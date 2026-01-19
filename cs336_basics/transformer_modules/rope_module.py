# import torch.nn as nn    
# import torch 
# class RoPE(nn.Module):
#     def __init__(self,
#                  theta: float,
#                  d_k:int,
#                  max_seq_len:int,
#                  device=None):
#         super().__init__()    
#         if d_k % 2 != 0:
#             raise ValueError(f"d_k must be even, got {d_k}")

#         self.theta = float(theta)
#         self.d_k = int(d_k)
#         self.max_seq_len = int(max_seq_len)
        
#         #一共有k对 (d_k/2 ,) (0,2,4,....)
#         k = torch.arange(0,d_k,2,device=device)
#         #一共有k对 (d_k/2 ,) (0,2,4,....)        
#         inv_freq = (self.theta ** (-k/d_k))
        
#         pos = torch.arange(self.max_seq_len,device=device)
        
#         # angles[i,k] = (i,1) * (1,k)
#         angles = pos[:,None] * inv_freq[None,:]
#         cos = torch.cos(angles)
#         sin = torch.sin(angles)

#         # Fixed buffers (not learnable, shared across layers)
#         self.register_buffer("cos_cached", cos, persistent=False)
#         self.register_buffer("sin_cached", sin, persistent=False)
        
#     def _rotate_half(self,x:torch.Tensor) -> torch.Tensor:
#         # x: (..., seq_len, d_k)
#         x_even = x[..., 0::2]
#         x_odd = x[..., 1::2]
#         rot = torch.stack((-x_odd, x_even), dim=-1)  # (..., seq_len, d_k/2, 2)
        
#         return rot.flatten(-2)                       # (..., seq_len, d_k)
        
#     def forward(self,
#                 x:torch.Tensor,
#                 token_positions:torch.Tensor)-> torch.Tensor:
#         # batch , seq_len
#         # 防御性编程 
#         token_positions = token_positions.to(x.device)
        
        
#         '''
#         (..., seq_len, 1)
#         ×
#         (1, d_k/2)
#         ↓
#         (..., seq_len, d_k/2)
#         '''
#         cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
#         sin = self.sin_cached[token_positions]  # (..., seq_len, d_k/2)
        
#         # cos/sin: (..., seq_len, d_k)
        
#         cos = cos.repeat_interleave(2, dim=-1).to(dtype=x.dtype)
#         sin = sin.repeat_interleave(2, dim=-1).to(dtype=x.dtype)

#         # apply: x*cos + rotate(x)*sin
#         return x * cos + self._rotate_half(x) * sin

import torch
import torch.nn as nn
import math

class RoPE(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k: int,
                 max_seq_len: int,
                 device=None):
        super().__init__()    
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")

        self.theta = float(theta)
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # 1. 使用 log 空间计算 inv_freq，增加数值稳定性
        # inv_freq 形状: (d_k/2,)
        # 公式: theta^(-2i/d_k)
        powers = torch.arange(0, d_k, 2, device=device).float() / d_k
        inv_freq = 1.0 / (self.theta ** powers)
        
        # 2. 预计算所有位置的 angles
        # pos 形状: (max_seq_len,)
        pos = torch.arange(self.max_seq_len, device=device)
        
        # 外积计算 angles: (max_seq_len, d_k/2)
        angles = torch.outer(pos, inv_freq)
        
        # 3. 预计算 cos 和 sin 并注册为 buffer
        # 形状都是 (max_seq_len, d_k/2)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        实现交替旋转: [x0, x1, x2, x3] -> [-x1, x0, -x3, x2]
        """
        # 确保内存连续，防止 flatten 导致数据乱序
        x = x.contiguous()
        
        x_even = x[..., 0::2] # x0, x2...
        x_odd = x[..., 1::2]  # x1, x3...
        
        # 拼接成 [..., d_k/2, 2] 然后展平
        rot = torch.stack((-x_odd, x_even), dim=-1)
        return rot.flatten(-2)
        
    def forward(self,
                x: torch.Tensor,
                token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (Batch, Head, Seq_len, d_head)
        token_positions: (Batch, Seq_len) 或 (1, Seq_len)
        """
        # 确保设备一致
        token_positions = token_positions.to(x.device)
        
        # 1. 根据 token_positions 提取对应的 cos/sin
        # 结果形状: (Batch, Seq_len, d_k/2)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # 2. 适配 4D Tensor (B, H, T, D)
        # 如果输入有 Head 维度，我们需要在 dim=1 插入一个维度以便广播
        if x.ndim == 4:
            cos = cos.unsqueeze(1) # (B, 1, T, d_k/2)
            sin = sin.unsqueeze(1) # (B, 1, T, d_k/2)
        
        # 3. 将 d_k/2 扩展回 d_k
        # 使用 repeat_interleave 配合交替旋转逻辑: [c0, c1] -> [c0, c0, c1, c1]
        cos = cos.repeat_interleave(2, dim=-1).to(dtype=x.dtype)
        sin = sin.repeat_interleave(2, dim=-1).to(dtype=x.dtype)

        # 4. 应用旋转矩阵公式
        # 
        return (x * cos) + (self._rotate_half(x) * sin)