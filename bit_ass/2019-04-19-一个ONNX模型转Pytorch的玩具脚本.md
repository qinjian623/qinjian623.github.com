---
layout: default
---
# 一个ONNX模型转Pytorch的玩具脚本

 [*Link:*](https://zhuanlan.zhihu.com/p/62963082)

这两天看ONNX的文档。这种数据格式的文档，大部分都是数据规范，完全没有动力看。

于是就弄个玩具吧，这样总要看一看吧。

  


虽然是self-contain，但是，稍微有点长，放在gist上了：

[ONNX2pytorch](https://gist.github.com/qinjian623/6aa777037534c1c1dccbb66f832e93b8)  




---

## 主要的问题  
基本都是数据载入、处理、转换，行行都XX是血泪，全是细心活，太锻炼我的少女心了。

感觉这是我写的血泪密度最高的脚本了。

## 简单测试  
我用pytorch.onnx转了torchvision中的res18/res50/dense121的模型到onnx文件，之后再用脚本转回DependencyModule，测试的时间和error在后面，时间肯定应该是慢了一点的。

这个error看着心里没谱，就当是我的代码没问题了吧...呵呵呵。

Don't touch, let's pretend it works.


```
Original: 0.14459800720214844
DependencyModule: 0.008149147033691406
Max error for res18.onnx : 0.0

Original: 0.016959428787231445
DependencyModule: 0.020435571670532227
Max error for res50.onnx : 0.0

Original: 0.018629789352416992
DependencyModule: 0.023691654205322266
Max error for dense121.onnx : 0.0
```
