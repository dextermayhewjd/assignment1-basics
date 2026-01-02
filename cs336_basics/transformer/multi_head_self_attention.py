import torch
import math
from einops import einsum,rearrange
from jaxtyping import Float
from torch import Tensor
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

    alpha = torch.softmax(scores, dim=-1)  # 对每一行做 softmax
    out = alpha @ V  # (S, D)
    return out
  

  
# ====== 多头 attention：======
def multi_head_self_attention(input_feature:Float[Tensor,"batch, seq_len, d_model"],
                              num_head:int,
                              W_Q:Float[Tensor,"batch, d_model, Hd_k"],
                              W_K:Float[Tensor,"batch, d_model, Hd_k"],
                              W_V:Float[Tensor,"batch, d_model, Hd_k"],
                              scale =True):
    """ 
    X : (seq_len,d_model)
    """
    Q = X @ W_Q   # (B, T, H*d_k)
    K = X @ W_K   # (B, T, H*d_k)
    V = X @ W_V   # (B, T, H*d_k)
    
    '''
    或者 
    '''
    Q = torch.einsum("b t d, d hd -> b t hd", X, W_Q)  # (B, T, H*d_k)
    K = torch.einsum("b t d, d hd -> b t hd", X, W_K)
    V = torch.einsum("b t d, d hd -> b t hd", X, W_V)
    
    B,T,_ = input_feature.shape
    H = num_head
    d_model = input_feature.shape[-1]
    d_k = d_model // H
    
    Qh = Q.view(B, T, H, d_k).transpose(1, 2)  # (B, H, T, d_k)
    Kh = K.view(B, T, H, d_k).transpose(1, 2)  # (B, H, T, d_k)
    Vh = V.view(B, T, H, d_k).transpose(1, 2)  # (B, H, T, d_k)
    
    Qh = rearrange(Q, "b t (h dk) -> b h t dk", h=H)  # (B, H, T, d_k)
    Kh = rearrange(K, "b t (h dk) -> b h t dk", h=H)
    Vh = rearrange(V, "b t (h dk) -> b h t dk", h=H)
    
    scores = (Qh @ Kh.transpose(-2, -1)) / math.sqrt(d_k)  # (B, H, T, T)
    scores = torch.einsum("b h t dk, b h s dk -> b h t s", Qh, Kh) / math.sqrt(d_k)
    
    attn = torch.softmax(scores, dim=-1)   # (B, H, T, T)
    
    Oh = attn @ Vh    # (B, H, T, d_k)
    Oh = torch.einsum("b h t s, b h s dk -> b h t dk", attn, Vh)

    O = Oh.transpose(1, 2).contiguous().view(B, T, H * d_k)  # (B, T, H*d_k)
    O = rearrange(Oh, "b h t dk -> b t (h dk)")  # (B, T, H*d_k)
