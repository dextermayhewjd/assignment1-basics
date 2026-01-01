import torch
import torch.nn as nn 
from einops import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math

from cs336_basics.transformer_modules.softmax_module import softmax

def scale_dot_product_attention(    
                Q: Float[Tensor, " batch_size  ...  seq_len  d_k"],
                K: Float[Tensor, " batch_size  ...  seq_len  d_k"],
                V: Float[Tensor, " batch_size  ...  seq_len  d_v"],
                mask: Bool[Tensor, " ... queries keys"] | None = None,
            ) -> Float[Tensor, " ... queries d_v"]:

        
        QtK = einsum(Q,K,"... queries d_k, ... keys d_k -> ... queries keys")
        d_k = Q.shape[-1]
        scale_Qtk = QtK/math.sqrt(d_k)
        
        masked_x = torch.where(mask,scale_Qtk,torch.tensor(float('-inf')))
        
        softmax_x = softmax(in_features=masked_x,dimension=-1)
        
        result = einsum(softmax_x,V, "... queries keys, ... keys d_v -> ... queries d_v")
        return result