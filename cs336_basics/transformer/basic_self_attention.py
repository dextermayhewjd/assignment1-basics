# 根据cs224n的 lecture 来的
'''
Docstring for cs336_basics.transformer.basic_self_attention
一个词的原始embedding 是 通过
one hot 来的

假设 W_1:n a sequence of words in vocabulary V ∈ ^ (d* V)
对于每个W_i (V,1)的大小 
    # hotshot(热编码 [0,1,0,0,....0]这种) 
x_i = EW_i  (d,V) * (V,1) = (d,1) 
    # 读取vocabulary里的对应token的embedding

1. 三个weight matrics QKV , 每个 R∈ d*d 
# 就是三个 (d,d)大小的矩阵 和token的embedding大小一样

q_i = Qx_i , k_i = Kx_i, v_i = Vx_i
# 都是计算后变成了 (d,d) * (d,1) 像是从QKV中读取了什么一样

2. 对于每对词都要算相似度
要算的那个词 是i的话 j是遍历所有词
e_ij = q_i^T * k_j 
a_ij = softmax(e_ij)

3.  计算output 每个词

o_i = Σ a_ij * v_i 
  
'''

import torch
# 这里假设就六个词
vocabulary = torch.randn((6,4))
print(vocabulary.shape)

token_hot_shot = torch.tensor([0,1,0,0])