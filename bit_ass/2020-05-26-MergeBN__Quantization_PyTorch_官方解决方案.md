---
layout: default
---
# MergeBN && Quantization PyTorch 官方解决方案

 [*Link:*](https://zhuanlan.zhihu.com/p/143664360)

最近一年PyTorch在部署相关的方向上有了很多发展。MergeBN 和 量化 这两个部署必备已经在官方支持内了。

自己的 MergeBN 实现是这个，不过官方存在实现后就没必要用这套了：

[Captain Jack：PyTorch 卷积与BatchNorm的融合](https://zhuanlan.zhihu.com/p/49329030)## Merge BN  
由于PyTorch的动态图特性，所以没有办法简单的实现智能合并（因为这个特性需要获得计算图，自己之前的计划是利用backword的跟踪来获取对应的Conv + BN 的组合。）

看了代码，官方实现方法基本和自己的实现等价（Dummy换成了Identity），但是为了准确，需要人工指定Conv， BN的名称。我当时偷懒就直接用的 ```module\_names()``` 输出的名称顺序，这个不能保证准确的匹配各种模型结构的。

## Quantization  
目前包括 qnnpack 和 fbgemm 两个后端

* qnnpack 只支持 per tensor
* fbgemm 支持 per channel

所以，fvgemm 肯定更准一点，不过只支持PC端。另外，我测了速度，fbgemm 相比 QNNPack 快了很多，应该是用了 PC CPU 上的 SSE 指令集。

### 注意事项  
* torchvision 里面的实现 residuel add 需要手动更改:


```
# In __init__:
        self.skip_add = nn.quantized.FloatFunctional()
# In forward:
        # out += identity
        out = self.skip_add.add(out, identity)
```
* 输入输出记得加Stub
* Inplace ReLU 不要 fuse
* qconfig 记得设置

## 结果  
在 ImageNet Val集合上的 torchvision res18 测试结果，量化数据集是 Image 1k 的 val 数据。


```
....................................................................................................................................................................................................
Before Q:
Evaluation accuracy 69.22

After Q:
Size of model after quantization
Size (MB): 11.719858
....................................................................................................................................................................................................
Evaluation accuracy 68.97
```
## 脚本  
[https://github.com/qinjian623/pytorch\_toys/blob/master/official\_quantization/main.py](https://github.com/qinjian623/pytorch\_toys/blob/master/official\_quantization/main.py)

