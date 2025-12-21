import torch
import math

torch.manual_seed(42)

vocabulary_dim = 256
embedding_dim = 512
vocabulary = torch.randn((vocabulary_dim, embedding_dim))

word_sequence_id = [1,3,5,255]
X = vocabulary[word_sequence_id]

W_q_matrix = torch.randn((embedding_dim,embedding_dim))
W_k_matrix = torch.randn((embedding_dim,embedding_dim))
W_v_matrix = torch.randn((embedding_dim,embedding_dim))


S, D = X.shape  # (seq_len,dimension)

# ====== 单头 attention：======
def single_head_self_attention(X, W_q, W_k, W_v, scale=True):
    """
    X: (S, D)
    return: (S, D)
    """
    Q = X @ W_q  # (S, D)
    K = X @ W_k  # (S, D)
    V = X @ W_v  # (S, D)

    scores = Q @ K.T  # (S, S)
    if scale:
        scores = scores / math.sqrt(D)

    alpha = torch.softmax(scores, dim=1)  # 对每一行做 softmax
    out = alpha @ V  # (S, D)
    return out
  

  
