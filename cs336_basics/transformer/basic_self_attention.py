# 根据cs224n的 lecture 来的
'''
Docstring for cs336_basics.transformer.basic_self_attention
一个词的原始embedding 是 通过
one hot 来的

假设 W_1:n a sequence of words in vocabulary V ∈ ^ (V,d)
对于每个W_i (1,V)的大小 
    # hotshot(热编码 [0,1,0,0,....0]这种) 
x_i = w_i @ E  (1,V) @ (V,d)   = (1,d) 
    # 读取vocabulary里的对应token的embedding

1. 三个weight matrics QKV , 每个 R∈ d*d 
# 就是三个 (d,d)大小的矩阵 和token的embedding大小一样

q_i = x_i@ Q , k_i = x_i@K, v_i = x_i @V
# 都是计算后变成了  (1,d)@ (d,d) = (1,d) 像是从QKV中读取了什么一样

2. 对于每对词都要算相似度
要算的那个词 是i的话 j是遍历所有词
e_ij = q_i * k_j ^T
a_ij = softmax(e_ij)

3.  计算output 每个词

o_i = Σ a_ij * v_j
  
'''

import torch
'''
固定seed 随机生成每次结果一样
'''
torch.manual_seed(42)
'''
这里假设就六个词 V= 256
每个词的embedding是 d = 16
'''
vocabulary_dim = 256 # 256个词
embedding_dim = 16 # 每个词 16 个维度

#(V,d)
vocabulary = torch.randn((vocabulary_dim,embedding_dim))

# (1,V) 
token_one_hot = torch.zeros((1,vocabulary_dim))

'''
按照要提取的词来从vocabul  的矩阵提取
eg 此处是提取第 18,5,223,36 个词
'''
word_sequence_id = [18,5,223,36] #  
sequence_length = len(word_sequence_id)
embedding = [] # n* embedding_dim
assert sequence_length == 4

''' 不用range 用 in'''
for i in word_sequence_id: 
    one_hot = token_one_hot.clone()
    '''
    (1, 256)
    ↑
    dimension 0 的 size = 1
    所以需要
    one_hot[0, i] = 1
    '''
    one_hot[0,i] = 1
    x_i = torch.mm(one_hot,vocabulary)   #(1,V) @ (V,d)
    embedding.append(x_i) #(1,d)
    
'''最快 
vocabulary: (V, d)
embedding = vocabulary[word_sequence_id]  
'''
embedding = torch.stack(embedding, dim=0).squeeze(1)  
# print(embedding.shape) # (d, seq_len)

W_q_matrix = torch.randn((embedding_dim,embedding_dim))
W_k_matrix = torch.randn((embedding_dim,embedding_dim))
W_v_matrix = torch.randn((embedding_dim,embedding_dim))


output = []
for i in range(sequence_length):
    x_i = embedding[i] # (1,d)
    q_i = x_i @ W_q_matrix # (1,d)
    
    scores = []
    for j in range(sequence_length):
        x_j = embedding[j]
        k_j = x_j @ W_k_matrix # (1,d) @ (d,d) 
        # print(k_j.shape)
        simliarity = torch.dot(q_i,k_j)
        '''
        q_i @ k_j  是点积 是变量 inner product
        '''
        scores.append(simliarity)
        
    scores = torch.stack(scores,dim=0) # (seq_len,)
    # print(scores.shape)
    # print(scores)
    alpha = torch.softmax(scores,dim = 0) # (seq_len,)
    # print(alpha.shape)
    # print(alpha)
    
    out_i = torch.zeros(embedding_dim)
    
    for j in range(sequence_length):
        x_j = embedding[j]
        v_j = x_j @ W_v_matrix # (1,d) @ (d,d) 
        out_i += alpha[j]*v_j #(1,d)

    '''如何累加求平均tensor'''
    output.append(out_i)
output = torch.stack(output)# (seq_len,d)

# print(output)
K = embedding @ W_k_matrix    # (seq_len, d)
V = embedding @ W_v_matrix    # (seq_len, d)
'''
这里的K 和 V 是针对于 已经堆叠的 embedding 算出来的  k和小v的叠加 
'''
output2 = []

for i in range(sequence_length):
    Q_i = embedding[i] @ W_q_matrix
    
    scores = []
    for j in range(sequence_length):
        # e_ij = Q_i @ K[j] #scalar 点积
        e_ij = torch.dot(Q_i,K[j])  
        '''就这里 (1,d) 和(1,d)  以及使用了 去原来的K的部分'''
        '''
        1. 这里是标准的 原先是 单条embedding(1,d) @ W_matrix_k 
            后来变成 embedding stack 后只取原本对应的 部分 —— * ||| = ——
                                                    
                                                    —————— * | | |     —————— （还是取原来的）
                                                    ——————   | | |  =  ——————
                                                    ——————   | | |     ——————
        
        2. 但其实这里还可以优化 for loop 加上两个张量tensor 求内积 得到的是标量 simliarity 
        那么其实可以通过直接乘整个矩阵 的转置T  
        
        
                                                    —————— * |      x   （内积 相似度）
                                                             |   = 
                                                             |     
        
                                                    ————— * | | |      x x x x x ( 一row的相似度 正好对应每qi 对于每个kj )
                                                            | | |  =  
                                                            | | |     
        
        
        '''
        
        scores.append(e_ij) 
    scores = torch.stack(scores) # (seq_len,)
    alpha = torch.softmax(scores,dim = 0) #(seq_len)
    
    out_i = alpha @V
    output2.append(out_i)
    
output2 = torch.stack(output2)# (seq_len,d)

assert torch.allclose(output, output2)
print(f'pass the test2')

output3 = []
for i in range(sequence_length):
    Q_i = embedding[i] @ W_q_matrix # (d,)
    scores = Q_i @ K.T # (seq_len)
    alpha = torch.softmax(scores,dim=0) # (seq_len)
    '''
    这里算出来的是 每个token的value 对应的比例
    '''
    output = alpha @ V # seq_len @ (seq_len,d)
    output3.append(output)
    
output3 = torch.stack(output3)# (seq_len,d)

assert torch.allclose(output3, output2)
print(f'pass the test3')



'''
第一行是 第一个token的q 对于所有key的 simliarity 
'''

Q = embedding @ W_q_matrix # (seq_len , d) @ (d,d) = （seq_len , d）
''' 可否理解为 一个embedding 就是从Q 矩阵中 得到 这个token的含义 通过每个dimension 上的比例 来翻译出 对应的 单个dimesnion上的意思'''
K = embedding @ W_k_matrix # （seq_len , d）
V = embedding @ W_v_matrix # （seq_len , d）

scores = Q@K.T # （seq_len , sequ_len）
alpha = torch.softmax(scores,dim = 1)
# 这一步 把每一行的simliarity 变成 alpha
output = alpha @ V  # （seq_len , sequ_len） @ （seq_len, d）
# 内积所以 1 @3 变成 3@3 型
# 第一行代表的 第一个词基于attention 得到value的表示
'''
Q 决定「我在问什么」
K 决定「我拥有什么信息可以被问」
V 决定「如果我被关注，我能提供什么内容」
'''