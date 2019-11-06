# 代码实现
一些错误的实现同样可以成功训练，模型能否收敛**不能**作为实现正确的标准。
## 必须人工静态检查
## 最好是互检/Code Review
## 不缺硬件的话，尽量不要用in place的操作

# 数据

# 训练

## 数据方向
Andrew的 machine learning yearning中提到了一些方法，比如如何判断
### 错误案例分析
### 判断数据集与网络规模



## 技术方向
### GAN辅助训练
### 多任务
### Finetune
### Warm Up
### Distilled
### 不要再follow传统的90 epoch的模式了，建议可以加倍，甚至更多。


## 技巧方向
### 从小数据开始
### 从小模型开始


## BN位置

## LR相关
1. https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html
2. https://arxiv.org/pdf/1803.09820.pdf
3. Warm Up


## Opt
### Adam
1. Adam （更加适合GAN），会有训练崩溃的问题。
2. https://www.fast.ai/2018/07/02/adam-weight-decay/ Adam的问题， amsgrad，AdamW
3. https://openreview.net/forum?id=ryQu7f-RZ

### SGDM
目前没有任何强有力的证据表明有任何比SGDM更好的优化器，所以，最终使用SGDM会是

# 网络结构
## 标配
1. Resisual
2. FPN
3. ReLU依然是最方便部署的
4. BatchNorm
5. 平衡好 Conv 的 Receptive Field
   + Deform
   + Dilated
   + Nonloca
## Extra
3. Attention



# 梯度消失


# 更快的收敛
## 三支柱问题
## Fast AI


# Ref：

https://arxiv.org/pdf/1708.07120.pdf
http://openaccess.thecvf.com/content_cvpr_2017/papers/Yim_A_Gift_From_CVPR_2017_paper.pdf
https://medium.com/neuralmachine/knowledge-distillation-dc241d7c2322
https://papers.nips.cc/paper/7338-how-to-start-training-the-effect-of-initialization-and-architecture.pdf
https://www.deeplearning.ai/ai-notes/initialization/
https://www.deeplearning.ai/ai-notes/optimization/