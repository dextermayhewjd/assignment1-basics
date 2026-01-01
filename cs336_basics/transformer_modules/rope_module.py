import torch.nn as nn    
import torch 
class RoPE(nn.Module):
    def __init__(self,
                 theta: float,
                 d_k:int,
                 max_seq_len:int,
                 device=None):
        super().__init__()    
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")

        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        
        #一共有k对 (d_k/2 ,) (0,2,4,....)
        k = torch.arange(0,d_k,2,device=device)
        #一共有k对 (d_k/2 ,) (0,2,4,....)        
        inv_freq = (self.theta ** (-k/d_k))
        
        pos = torch.arange(self.max_seq_len,device=device)
        
        # angles[i,k] = (i,1) * (1,k)
        angles = pos[:,None] * inv_freq[None,:]
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        # Fixed buffers (not learnable, shared across layers)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
        
    def _rotate_half(self,x:torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot = torch.stack((-x_odd, x_even), dim=-1)  # (..., seq_len, d_k/2, 2)
        
        return rot.flatten(-2)                       # (..., seq_len, d_k)
        
    def forward(self,
                x:torch.Tensor,
                token_positions:torch.Tensor)-> torch.Tensor:
        # batch , seq_len
        
        '''
        (..., seq_len, 1)
        ×
        (1, d_k/2)
        ↓
        (..., seq_len, d_k/2)
        '''
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # (..., seq_len, d_k/2)
        
        
        # cos/sin: (..., seq_len, d_k)
        
        
        cos = cos.repeat_interleave(2, dim=-1).to(dtype=x.dtype)
        sin = sin.repeat_interleave(2, dim=-1).to(dtype=x.dtype)

        # apply: x*cos + rotate(x)*sin
        return x * cos + self._rotate_half(x) * sin