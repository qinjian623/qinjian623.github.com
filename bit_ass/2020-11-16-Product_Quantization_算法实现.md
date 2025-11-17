---
layout: default
---
# Product Quantization 算法实现

 [*Link:*](https://zhuanlan.zhihu.com/p/296841635)

看了下自己知乎的文章，再不抓紧随便弄一篇，就搞不定自己的私人KPI了。为了 KPI ，水一份也是一份。

最近工作上要用 PQ 算法，本来 Faiss 是有的，但是我懒得看代码怎么把结果提出来了，而且自己也需要对 PQ 的所有参数都能有控制能力。反正 k-means 已经有现成的了，顺手的事情。

目前已经在私有的数据集上做了测试，符合预期。使用 PQ 的场景下，初步的测试表明，无论是速度上的提升，还是内存占用上的优化，都达到了预期，后面应该会进一步用规模大一点的数据进行测试。

测试/示例代码位置：

[https://github.com/qinjian623/pytorch\_toys/blob/master/cluster/pq\_test.py](https://github.com/qinjian623/pytorch\_toys/blob/master/cluster/pq\_test.py)上面的简化实际流程：


```
codebook = product_quantization(training_data, subv_size, k=num_centers)     # 训练 PQ 获得类中心
pq_data = data_to_pq(data, codebook)      # 量化数据
tb = asymmetric_table(query, codebook)   # 获得 query 的非对称距离查询表
for batch in torch.split(pq_db, 2048): 
        distances.append(asymmetric_distance(tb, batch))    # batch size = 2048，计算非对称距离
distances = torch.cat(distances, dim=1)     # 合并 batch
score, ids = distances.topk(k, dim=1, largest=False)              # 获得 topk 距离结果和 id
```
随机测试的结果：


```
Precomputing: 0.0020322799682617188s
Distancing: 0.012085199356079102s
Recall: 0.99 [M=8, bits=8.0 @ top 32]
```
  


立个 Flag，后续要写点 NN 训练相关的。

