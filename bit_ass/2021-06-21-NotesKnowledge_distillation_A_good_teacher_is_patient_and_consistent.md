---
layout: default
---
# [Notes]Knowledge distillation: A good teacher is patient and consistent

 [*Link:*](https://zhuanlan.zhihu.com/p/382643408)

模型加速三板斧：

* 量化
* 减枝
* distill

这篇论文做的是 distillation 相关的试验验证，而且是 Google 不差钱风。

这篇论文的好处是，没任何额外的 trick ，拿来就可以用。看别人烧钱在大规模数据上的结论，省自己的钱。


> The apparent simplicity of our findings

两个 principles ：

1. Teacher and student should process the exact same input image viewsor, more specifically, same crop and augmentations.
2. We want the functions to match on alarge number of support points to generalize well.


> With this in mind, we experimentally demonstrate that consistent image views, aggressive augmentations and very long training schedulesare the key to makemodel compression via knowledge distillationwork well in practice.

论文中一些关于 distillation 的相关结论：

1. 数据增强要保证 teacher(T), student(S) 两者一致。
2. 训练 epoch 要增大，文中的直接用的 10k epoch，没有发生 overfit（没有 overfit，那就是 deep learning 的天堂）。
3. 可以用高清输入的 T 来训练 S，但是看文中的精度，提升性价比不高，要消耗太多算力。
4. Pretrained 的模型反而不如直接 from scratch（也不意外，现在很多场景 pretrained 其实作用不大了，尤其现在的 AI 公司，私有数据规模都已经很大了，毕竟 pretrained 的也只是一套“还可以”的参数）。
5. 异构的模型一样可以直接 distill，mobilenet、resnet 随便训。
6. OOD 的数据之间 distill 效果不理想（这也不意外，毕竟不是通用特征，但是如果你有个"上帝"数据集，里面什么都有，那肯定照样有效果）。



---

**自己的民科**

1. 有时候简单粗暴比一堆 trick 有效多了。在 deep learning 里，可能之前的很多工作随着数据的增加都会变得失去意义。
2. 在工业界，其实 distill 比单纯的追 sota 算法更加重要，毕竟 distill 是个训练框架。一些 sota 算法为了刷指标可能有一些场景限制、不考虑实用等等。何况，用了 distill，一样可以尝试把 sota 模型蒸馏到自己朴素的 resnet、Faster-RCNN、Yolo上。
3. 对于工业界训练，数据规模上到 m 级别以上后，就可以开始训专家模型了。专家模型后面也可以用 ensemble、cascade子分类器来进一步通过蒸馏来指导线上模型的训练。
