---
layout: default
---
# Darknet yolo_layer实现

 [*Link:*](https://zhuanlan.zhihu.com/p/36961935)

由于之前总结性的看了yolo系列的3篇，这两天好奇具体的yolo实现，于是看了看darknet有关的具体代码。darknet整体的CPU的代码直白清晰，挺好看的，虽然可能缺少一些小技巧的优化，但是都用CPU了，谁在意这个。

  


## 程序入口  
  



> examples/darknet.c

  


官网上的训练命令是

  



> ./darknet detector train cfg/voc.data cfg/yolov3-voc.cfg darknet53.conv.74

  


顺路继续找下去的位置就是

  



> examples/detector.c  
> 里面的：

  



```
void train_detector(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear);

```
  


当然，其实上面也没什么卵用。反正直接看cfg里面的文件，可以知道用的是yolo层。可以在parse.c里面看到实际用的就是src/yolo\_layer.c里面的实现。

  


## 内存分配  
  


直接加注释了，有些配置都是在cfg文件，还有parser里面可以看到。

  



```
layer make_yolo_layer(int batch, int w, int h, int n, int total, int *mask, int classes)
{
    int i;
    layer l = {0};
    l.type = YOLO;
    l.n = n; // 这一层用的anchor数量
    l.total = total; // 所有的anchor数量
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = n*(classes + 4 + 1); // classe, bbox, objectness
    l.out_w = l.w;
    l.out_h = l.h;
    l.out_c = l.c;
    l.classes = classes;
    l.cost = calloc(1, sizeof(float));
    l.biases = calloc(total*2, sizeof(float)); // anchor的具体值
    // below 具体使用的那几个anchor
    if(mask) l.mask = mask;
    else{
        l.mask = calloc(n, sizeof(int));
        for(i = 0; i < n; ++i){
            l.mask[i] = i;
        }
    }
    l.bias_updates = calloc(n*2, sizeof(float));
    l.outputs = h*w*n*(classes + 4 + 1);
    l.inputs = l.outputs;
    l.truths = 90*(4 + 1);
    l.delta = calloc(batch*l.outputs, sizeof(float)); // MSE的差
    l.output = calloc(batch*l.outputs, sizeof(float));
    for(i = 0; i < total*2; ++i){
        l.biases[i] = .5;
    }

    l.forward = forward_yolo_layer;
    l.backward = backward_yolo_layer;
#ifdef GPU
    l.forward_gpu = forward_yolo_layer_gpu;
    l.backward_gpu = backward_yolo_layer_gpu;
    l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
    l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

    fprintf(stderr, "yolo\n");
    srand(0);

    return l;
}

```
## 前向  
两个循环。

首先，网络的每个输出的bbox都对比groudtruth，如果IOU > ignore则不参与训练，进一步的，大于truth则计算loss，参与训练，但是cfg文件中这个值设置的是1,所以应该就是忽略后面这个进一步的了。

第二个循环，对每个目标，查找最合适的anchor，如果本层负责这个尺寸的anchor，就计算对应的各loss。否则忽略。

  



```
box get_yolo_box(float *x, float *biases, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
    box b;
    // lw/lh为网络输出大小， b.x, b.y 为全图相对尺寸
    b.x = (i + x[index + 0*stride]) / lw;
    b.y = (j + x[index + 1*stride]) / lh;
    // w/h为网络输入大小， bias为anchor尺寸，b.w, b.h为全图相对尺寸
    b.w = exp(x[index + 2*stride]) * biases[2*n]   / w;
    b.h = exp(x[index + 3*stride]) * biases[2*n+1] / h;
    return b;
}

// 计算boundbox的loss
float delta_yolo_box(box truth, float *x, float *biases, int n, int index,
                 int i, int j, int lw, int lh, int w, int h, float *delta,
                 float scale, int stride)
{
    box pred = get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride);
    float iou = box_iou(pred, truth);
    float tx = (truth.x*lw - i);
    float ty = (truth.y*lh - j);
    float tw = log(truth.w*w / biases[2*n]);
    float th = log(truth.h*h / biases[2*n + 1]);


    // scale = 2 - truth.w * truth.h 干毛线的？ 
    delta[index + 0*stride] = scale * (tx - x[index + 0*stride]);
    delta[index + 1*stride] = scale * (ty - x[index + 1*stride]);
    delta[index + 2*stride] = scale * (tw - x[index + 2*stride]);
    delta[index + 3*stride] = scale * (th - x[index + 3*stride]);
    return iou;
}

// 计算分类loss
void delta_yolo_class(float *output,
                    float *delta,
                    int index,
                    int class,
                    int classes,
                    int stride,
                    float *avg_cat)
{
    int n;
    // delta[index] is not 0. one class?
    if (delta[index]){
        delta[index + stride*class] = 1 - output[index + stride*class];
        if(avg_cat) *avg_cat += output[index + stride*class];
        return;
    }
    // multi-class
    for(n = 0; n < classes; ++n){
        delta[index + stride*n] = ((n == class)?1 : 0) - output[index + stride*n];
        if(n == class && avg_cat) *avg_cat += output[index + stride*n];
    }
} 
void forward_yolo_layer(const layer l, network net)
{
    int i,j,b,t,n;
    memcpy(l.output, net.input, l.outputs*l.batch*sizeof(float));

#ifndef GPU
    for (b = 0; b < l.batch; ++b){
        for(n = 0; n < l.n; ++n){
            int index = entry_index(l, b, n*l.w*l.h, 0);
            activate_array(l.output + index, 2*l.w*l.h, LOGISTIC); // obj
            index = entry_index(l, b, n*l.w*l.h, 4);
            activate_array(l.output + index, (1+l.classes)*l.w*l.h, LOGISTIC); // classes
        }
    }
#endif

    memset(l.delta, 0, l.outputs * l.batch * sizeof(float));
    if(!net.train) return;

    // 以上是测试
    float avg_iou = 0;
    float recall = 0;
    float recall75 = 0;
    float avg_cat = 0;
    float avg_obj = 0;
    float avg_anyobj = 0;
    int count = 0;
    int class_count = 0;
    *(l.cost) = 0;
    for (b = 0; b < l.batch; ++b) {
        for (j = 0; j < l.h; ++j) {
            for (i = 0; i < l.w; ++i) {
                for (n = 0; n < l.n; ++n) {
                    int box_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 0);
                    box pred = get_yolo_box(l.output, l.biases, l.mask[n],
                                            box_index, i, j,
                                            l.w, l.h,
                                            net.w, net.h,
                                            l.w*l.h);
                    float best_iou = 0;
                    int best_t = 0;
                    for(t = 0; t < l.max_boxes; ++t){
                        box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
                        if(!truth.x) break;
                        float iou = box_iou(pred, truth);
                        if (iou > best_iou) {
                            best_iou = iou;
                            best_t = t;
                        }
                    }
                    // obj
                    int obj_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4);
                    avg_anyobj += l.output[obj_index];
                    // Negative
                    l.delta[obj_index] = 0 - l.output[obj_index];
                    // 如果大于igonre_thresh就不参与训练
                    if (best_iou > l.ignore_thresh) {
                        l.delta[obj_index] = 0;
                    }
                    // 进一步大于truth_thresh，参与训练
                    if (best_iou > l.truth_thresh) {
                        l.delta[obj_index] = 1 - l.output[obj_index];

                        // groudtruth的类型
                        int class = net.truth[best_t*(4 + 1) + b*l.truths + 4];
                        if (l.map) class = l.map[class];
                        // 网络预测类型的index
                        int class_index = entry_index(l, b, n*l.w*l.h + j*l.w + i, 4 + 1);
                        delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, 0);
                        box truth = float_to_box(net.truth + best_t*(4 + 1) + b*l.truths, 1);
                        delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index,
                                    i, j, l.w, l.h, net.w, net.h, l.delta,
                                    (2-truth.w*truth.h), l.w*l.h);
                    }
                }
            }
        }
        // 保证所有的目标都应该出现在预测中。
        for(t = 0; t < l.max_boxes; ++t){
            box truth = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);

            if(!truth.x) break;
            float best_iou = 0;
            int best_n = 0;
            i = (truth.x * l.w);
            j = (truth.y * l.h);
            box truth_shift = truth;
            truth_shift.x = truth_shift.y = 0;
            // 搜索最合适的anchor
            for(n = 0; n < l.total; ++n){
                box pred = {0};
                pred.w = l.biases[2*n]/net.w;
                pred.h = l.biases[2*n+1]/net.h;
                float iou = box_iou(pred, truth_shift);
                if (iou > best_iou){
                    best_iou = iou;
                    best_n = n;
                }
            }
            int mask_n = int_index(l.mask, best_n, l.n);
            // 如果最合适的anchor由本层负责预测（由mask来决定）
            if(mask_n >= 0){
                // 类似上面的工作。
                int box_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 0);
                float iou = delta_yolo_box(truth, l.output, l.biases,
                                        best_n, box_index,
                                        i, j, l.w, l.h, net.w, net.h,
                                        l.delta,
                                        (2-truth.w*truth.h),
                                        l.w*l.h);

                int obj_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4);
                avg_obj += l.output[obj_index];
                l.delta[obj_index] = 1 - l.output[obj_index];

                int class = net.truth[t*(4 + 1) + b*l.truths + 4];
                if (l.map) class = l.map[class];
                int class_index = entry_index(l, b, mask_n*l.w*l.h + j*l.w + i, 4 + 1);
                delta_yolo_class(l.output, l.delta, class_index, class, l.classes, l.w*l.h, &avg_cat);

                ++count;
                ++class_count;
                if(iou > .5) recall += 1;
                if(iou > .75) recall75 += 1;
                avg_iou += iou;
            }
        }
    }
    // MSE loss
    *(l.cost) = pow(mag_array(l.delta, l.outputs * l.batch), 2);
    printf("Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d\n", net.index, avg_iou/count, avg_cat/class_count, avg_obj/count, avg_anyobj/(l.w*l.h*l.n*l.batch), recall/count, recall75/count, count);
}

```
  


## 后向  
  


就一句， 相当于直接拷贝之前的delta了。

  



```
axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, net.delta, 1);
```
