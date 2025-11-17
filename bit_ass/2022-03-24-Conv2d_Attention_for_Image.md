---
layout: default
---
# Conv2d Attention for Image

 [*Link:*](https://zhuanlan.zhihu.com/p/487058629)

我自己用来做 2D image 之间的 Attention 的玩具，把 Linear 换成了 Conv2d，免得还得纬度转来转去，其他没变。上一个版本有过任务验证，起码能收敛。

[https://github.com/qinjian623/pytorch\_toys/blob/master/utils/modules.py](https://github.com/qinjian623/pytorch\_toys/blob/master/utils/modules.py)  



```
class Conv2dCrossAttention(nn.Module):
    r"""
    Attention from 2D vk space to 2D query space. If they are the same, this module will be a self-attention.

    VK space: (bs, D_vk_space, vk_h, vk_w)

    Q space: (bs, D_query_space, query_h, query_w)

    Output: (bs, D_output, query_h, query_w)
    """

    def __init__(self,
                 D_query_space: int,
                 D_vk_space: int,
                 D_emb: int,
                 D_output: int):
        super(Conv2dCrossAttention, self).__init__()
        self.query = nn.Conv2d(D_query_space, D_emb, 1)
        self.key = nn.Conv2d(D_vk_space, D_emb, 1)
        self.value = nn.Conv2d(D_vk_space, D_output, 1)
        self.scale = sqrt(D_emb)
        self.init()  # Special init.

    def init(self):
        # Without ReLU, xavier is better than kaiming, maybe.
        xavier_uniform_(self.query.weight)
        xavier_uniform_(self.key.weight)
        xavier_uniform_(self.value.weight)

    def forward(self, image: Tensor, query_space: Tensor):
        # Boring QKV
        query = self.query(query_space)
        key = self.key(image)
        value = self.value(image)

        bs, bc, bh, bw = query_space.shape
        _, ic, ih, iw = image.shape

        # Reshape to MLP style
        # Eliminate spatial dim
        # Permute to (batch size, seq length, emb) x (bs, emb, seq)
        query = query.reshape(bs, -1, bh * bw)
        key = key.reshape(bs, -1, ih * iw).permute(0, 2, 1)
        value = value.reshape(bs, -1, ih * iw)

        # Scores and outputs
        scores = torch.bmm(key, query) / self.scale
        weights = F.softmax(scores, dim=-2)

        # Weighted sum of value.
        outputs = torch.bmm(value, weights)

        # Back to CNN style
        outputs = outputs.reshape(bs, -1, bh, bw)
        return outputs
```
