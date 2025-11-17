---
layout: default
---
# Distillation gayhub上的实现 bug

 [*Link:*](https://zhuanlan.zhihu.com/p/197965267)

自己的玩具训练要弄个distiller，于是 Google 了下，反正是第一条： 

[https://github.com/peterliht/knowledge-distillation-pytorch](https://github.com/peterliht/knowledge-distillation-pytorch)用自己的模型放进去测了一下，看着似乎也没问题，于是先把 Loss 的计算那块挪到自己的代码里面。随后多次训练总是感觉很诡异，于是才进去看了这个项目的代码。

实际上，按照这个项目最新版本代码，就根本没有 Distillation。作者提前生成了 Cifar10 的 Teacher Output，训练时是从这些结果里面直接取结果的。

作者应该是希望这样能够减少大模型的运行次数，结果直接带来两个bug：

1. 每张图片在不同 epoch 下的 augmentation 都是随机变化的，这已经和最先生成时的数据不一致了。（这个或许还能有奇效）
2. 每个 epoch 都要重新打乱数据集，所以 teacher output 在训练中根本就不是对应的图片，distill 的实际是随机某一张图片的 teacher output。（讲真，我后来想了想，这tm也能有奇效）

实际上项目的 issue 里面已经有很多人提了这些疑惑。我相信作者本身提供的模型也是正确代码版本下获得的，但是几个版本、脑洞改下来，代码自然引入了bug。

那么问题来了，为什么训练还显得如此正常？？？

这就回到“奇效”的两个问题上来：

### A.数据增强带来的 bug  
每次 augmentation 虽然变化了，但是即使不用 distillation，label 也是固定的，所以，可以认为这个bug的效果是：teacher 针对数据集生成了 logits 的 label，而且不论这个图像经过了如何变换，student 都应该与这个 label 保持一致。

没毛病，就是一个针对模型输出的 constrain，可以有增加泛化的潜力。由于图像变化不会对这 teacher output 有影响，所以会强迫 student 学习对图像变化的抗干扰能力。

所以，如果 teacher output 是按照测试时的预处理设定获得的结果，会不会反而有奇效？

### B. 数据集 shuffle 的 bug  
这个bug的效果就是：每次用一个完全无厘头的随机数据对 student 的输出做限制。这也是实际训练中展示出来的效果。

当然，我这里测试的结果是略微的负面效果。



---

虽然有bug，但是模型还是能够正常训练，差异也很小。去查代码也全靠自己的直觉。基本就是 DL 代码面临的老大难问题：

1. 难以回测，一个大点的模型训下来，都是按小时、按天计数。改改代码，要回测的话，又得相同时间来确认。
2. 错误无法暴露，代码上稀里糊涂出点问题模型也能正常训练，甚至看不出反常。

  


后来我找到的另外的一个实现其实也有小问题： 

[https://github.com/moskomule/distillation.pytorch/blob/bfc92600092c12dac42c9fc5d4c199c60a5987f5/hinton/utils.py#L37](https://github.com/moskomule/distillation.pytorch/blob/bfc92600092c12dac42c9fc5d4c199c60a5987f5/hinton/utils.py#L37)这里没有 CELoss 的加权，不过问题不大。

应该正确的实现： 

[https://github.com/NervanaSystems/distiller/blob/94af2955f99de8222bd83c1fc46f4000b3ecb130/distiller/knowledge\_distillation.py#L149-L164](https://github.com/NervanaSystems/distiller/blob/94af2955f99de8222bd83c1fc46f4000b3ecb130/distiller/knowledge\_distillation.py#L149-L164)还是商业公司的产品好啊。

我自己也实现了一版， 不想一个小玩具还要引入一堆依赖：

[https://github.com/qinjian623/pytorch\_toys/blob/master/loss/distill.py](https://github.com/qinjian623/pytorch\_toys/blob/master/loss/distill.py)## Ref:  
1. TopK KD: [https://arxiv.org/pdf/2002.03532.pdf](https://arxiv.org/pdf/2002.03532.pdf)
2. The State Of Knowledge Distillation For Classification Tasks: [https://arxiv.org/pdf/1912.10850.pdf](https://arxiv.org/pdf/1912.10850.pdf)的加权
3. Original: [https://arxiv.org/pdf/1503.02531.pdf](https://arxiv.org/pdf/1503.02531.pdf)
