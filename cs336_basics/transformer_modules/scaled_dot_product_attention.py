import torch
import torch.nn as nn 
from einops import einsum
from jaxtyping import Bool, Float, Int
from torch import Tensor
import math

from cs336_basics.transformer_modules.softmax_module import softmax

# def scale_dot_product_attention(    
#                 Q: Float[Tensor, " batch_size  ...  seq_len  d_k"],
#                 K: Float[Tensor, " batch_size  ...  seq_len  d_k"],
#                 V: Float[Tensor, " batch_size  ...  seq_len  d_v"],
#                 mask: Bool[Tensor, " ... queries keys"] | None = None,
#             ) -> Float[Tensor, " ... queries d_v"]:

        
#         QtK = einsum(Q,K,"... queries d_k, ... keys d_k -> ... queries keys")
#         d_k = Q.shape[-1]
#         scale_Qtk = QtK/math.sqrt(d_k)
        
#         # if mask is not None:
#         #     scores = torch.where(
#         #         mask,
#         #         scores,
#         #         # torch.tensor(float('-inf'))
#         #         torch.full_like(scores, float("-inf")),
#         #         )
#         scores = scale_Qtk
#         if mask is not None:
#             scores = scores.masked_fill(~mask, float("-inf"))

#         softmax_x = softmax(in_features=scores, dimension=-1)
        
#         result = einsum(softmax_x,V, "... queries keys, ... keys d_v -> ... queries d_v")
#         return result
    
    
def scale_dot_product_attention(    
        Q: Float[Tensor, "batch ... queries d_k"],
        K: Float[Tensor, "batch ... keys d_k"],
        V: Float[Tensor, "batch ... keys d_v"],
        mask: Bool[Tensor, "... queries keys"] | None = None,
    ) -> Float[Tensor, "... queries d_v"]:
        
        d_k = Q.shape[-1]
        
        # 1. 计算缩放的点积得分
        # 使用 einsum 自动处理 Batch 和 Head 维度
        scores = einsum(Q, K, "... q d_k, ... k d_k -> ... q k") / math.sqrt(d_k)
        
        # 2. 应用掩码
        if mask is not None:
            # 确保 mask 在同一个设备上，且取反 (~mask) 将 True(1) 变为 False(0)
            # 这样 mask 中为 False 的地方会被填充为 -inf
            scores = scores.masked_fill(~mask, float("-inf"))

        # 3. Softmax
        # 建议此处先用官方实现排查问题，如果过拟合成功，再换回自定义 softmax
        attn_probs = torch.nn.functional.softmax(scores, dim=-1)
        
        # 4. 加权求和
        # (..., q, k) @ (..., k, d_v) -> (..., q, d_v)
        result = einsum(attn_probs, V, "... q k, ... k d_v -> ... q d_v")
        
        return result