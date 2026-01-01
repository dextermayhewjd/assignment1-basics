import torch.nn as nn    
import torch 
class rope():
    def __init__(self,
                 theta: float,
                 d_k:int,
                 max_seq_len:int,
                 device=None):
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even, got {d_k}")

        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)
        
        #一共有k对 (d_k/2 ,) (0,2,4,....)
        k = torch.arange(0,d_k,2,device=device)
        
        #一共有k对 (d_k/2 ,) (0,2,4,....)        
        inv_freq = (self.theta ** (-k/d_k))


    def _rotate_half(x:torch.Tensor) -> torch.Tensor:
        # x: (..., seq_len, d_k)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rot = torch.stack((-x_odd, x_even), dim=-1)  # (..., seq_len, d_k/2, 2)
        
        return rot.flatten(-2)                       # (..., seq_len, d_k)
        
    def forward(self,
                x:torch.Tensor,
                token_positions:torch.Tensor)-> torch.Tensor:
        # batch , seq_len
        token_positions.to(dtype=self.inv_freq.dtype)
        
        # (...,seq_len) - (..., seq_len, 1)
        angles =( token_positions[..., None] 
                    * self.inv_freq[None, :]  )
        '''
        (..., seq_len, 1)
        ×
        (1, d_k/2)
        ↓
        (..., seq_len, d_k/2)
        '''
        # cos/sin: (..., seq_len, d_k)
        cos = angles.cos().repeat_interleave(2, dim=-1).to(dtype=x.dtype)
        sin = angles.sin().repeat_interleave(2, dim=-1).to(dtype=x.dtype)

        # apply: x*cos + rotate(x)*sin
        return x * cos + self._rotate_half(x) * sin