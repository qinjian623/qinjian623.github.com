---
layout: default
---
# Notes on Neural Machine Translation by Jointly Learning to Align and Translate

 [*Link:*](https://zhuanlan.zhihu.com/p/28515057)

这是大概三或者两个月前无聊的时候看的几篇有关Attenion Model的论文之一。那段时间正好有一周轮到我在组会上讲东西，不知道哪里出的毛病突然就搜了几篇这个主题的文章。因为是一周读了大概七八篇，所以对我来说算是速度很快，自然细节会遗漏很多。现在凭记忆再整理成文字，也就只能讲讲所谓的思想了，一般来说讲思想就可以糊弄糊弄。

  


首先，这篇是针对seq2seq的改进。seq2seq不多说，基本可以认为是先塞一个句子给LSTM跑一遍，完事了拉出来中间的隐藏层的结果，再用一个LSTM跑一遍，出最终结果，这个基本思想就是Encoder-Decoder。只是，我相信正常人第一次看完这篇文章心里一定是“这TM也行”的感叹，因为，句子不一样长。不一个长度的句子复杂度肯定不一样，seq2seq完全没考虑这个问题。不过，数据集上的结果非常好，你能奈何呢。

  


这篇论文就是试图解决这个变长问题，可以猜测作者的讨论对话：

A: "seq2seq这篇实在太不科学了。"

B: "没错。"

A: "这说明了什么？"

B: "模型面前，科学根本不重要。"

A: "滚，说明了我们可以找到一个更科学的方法，在seq2seq的基础上获得更好的结果。"

  


于是，最朴素的想法是什么？在Decoding的阶段，把Encoder LSTM的运行过程中所有的的隐藏层结果利用起来。所以第一个动作应该就 $\sum_{t=0}^N{h_t}$ 。当然，单这样的话最后的和的范围会变化，所以肯定会变成 $1/N * \sum_{t=0}^N{h_t}$ 。我个人猜测在这个基础上的结果已经有改进了，不过这是个单纯的平均。进一步的改进显然就是加权平均，这样就需要一个选择权值的函数 $F_w$ 。

  


为了定义这个 $F_w$ ，需要找到输入和输出，输出显然就是一个权值，输出则需要看看当时情况下可以用的数据包括哪些。在Decoding的每一个step，可以拿来用的数据就包括：上一步的结果 $s_{t-1}$ ，Encoding过程中的所有中间结果。于是，我们可以定义 $F_w = f(s_{t-1}, h_j)$ 。文章里面说的貌似比较细致一点，其实就是一个输入为 $s_{t-1}, h_j$ ，输出为一个权值的网络，用来拟合这个 $F_w$ 。之后会把对每个h的所有的结果再弄一个Softmax，其实就是为了做normalize。不过，其实这个和另外一个东西很类似，LSTM里面的gate。

  


当然，我的水平有限，表达更是灾难。

  


Refs:

* [http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)
* [https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

