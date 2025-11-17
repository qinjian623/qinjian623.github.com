---
layout: default
---
# Naive im2col

 [*Link:*](https://zhuanlan.zhihu.com/p/33850484)

还是年前最后一天写的玩具，原计划准备过年的时候把后向也实现了。

然而，人性是那么的强大，呵呵。我看着PS4上刚买的一堆游戏，就已经知道故事的结局了，就这样吧。

  


参照了chainer的实现。im2col\_by\_kern 这个要快不少。用chainer实现对比测试了结果，应该没错。

  


市面上不少框架，适合看代码的，个人观点是抽象一点可以看Chainer的Python，底层一点的可以看caffe的cpp。两者代码都组织的不错，结构比较清晰，按理说caffe2应该组织的也很好，不过我没自己看过，没有发言权。

  



```
import cv2
import numpy as np
import time
from math import ceil


def conv_output_axis(l, k, s, p):
    padding_l = l + 2*p
    # equal to (padding_l - k)//s + 1
    # don't forget float in 2.x, FXXX --!
    return ceil((padding_l - (k - 1))/float(s))


def conv_output_size(img, kernel, stride, padding):
    n, c, h, w = img.shape
    kh, kw = kernel
    sh, sw = stride
    ph, pw = padding
    oh = conv_output_axis(h, kh, sh, ph)
    ow = conv_output_axis(w, kw, sw, pw)
    return (int(oh), int(ow))


# n, c, h, w => n, c, kh, kw, oh, ow
# or
# n, c, h, w => n, (c* kh* kw), oh, ow
# TODO too many loops
def im2col(img, k, s, p, pval=0):
    output_size = conv_output_size(img, k, s, p)
    oh, ow = output_size
    ph, pw = p
    kh, kw = k
    sh, sw = s
    n, c, h, w = img.shape
    # Pad
    img = np.pad(img,
                 ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                 mode='constant',
                 constant_values=pval)
    col = np.ndarray((n, c * kh * kw, oh, ow),
                     dtype=img.dtype)
    kern_weights_size = [n, c*kh*kw]
    for h in range(oh):
        hs, he = h*sh, h*sh+kh
        for w in range(ow):
            col[:, :, h, w] = img[:, :,
                                  hs: he,
                                  w*sw: w*sw + kw].reshape(
                                      kern_weights_size)
    return col


# Less loops version, much faster.
def im2col_by_kern(img, k, s, p, pval=0):
    output_size = conv_output_size(img, k, s, p)
    oh, ow = output_size
    ph, pw = p
    kh, kw = k
    sh, sw = s
    n, c, h, w = img.shape
    # Pad
    img = np.pad(img,
                 ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                 mode='constant',
                 constant_values=pval)
    col = np.ndarray((n, c * kh * kw, oh, ow),
                     dtype=img.dtype)
    flat_kern_size = c*kh*kw
    for h in range(kh):
        hs, he = h, h + sh * oh
        for w in range(kw):
            # n, c*k^2, h, w
            col[:, kw*h+w: flat_kern_size: kh*kw, :, :] = img[:,
                                                              :,
                                                              hs:he:sh,
                                                              w:w + sw * ow:sw]
    return col


def conv_forward(img, W, b, k, s, p, pval=0):
    cols = im2col_by_kern(img, k, s, p, pval)
    # cols => n, c * kh * kw, oh, ow
    # W    => out_c, c * kh * kw
    # bias => 1, c_out
    y = np.tensordot(cols, W, (1, 1))
    if b is not None:
        y += b
    y = np.transpose(y, (0, 3, 1, 2))
    return y
```
