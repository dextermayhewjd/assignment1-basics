import torch
import torch.nn as nn
from einops import einsum

from cs336_basics.transformer_modules.linear_module import Linear
class SwiGLU_feed_forward(nn.Module):
    def __init__(self,
                 d_model:int,
                 d_ff:int 
                 ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        # self.d_ff = (8/3)*self.d_model
        # self.d_ff = int(8 * d_model / 3)
                
        self.linear_1 = Linear(out_features=self.d_ff,in_features=self.d_model)
        self.linear_2 = Linear(out_features=self.d_model,in_features=self.d_ff)
        self.linear_3 = Linear(out_features=self.d_ff,in_features=self.d_model)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        part1 = self.linear_1(x)
        part3 = self.linear_3(x)
        
        silu_part1 = torch.sigmoid(part1) * part1
        
        
        mid_part = einsum(silu_part1,part3,
                  "batch sequence out_f ,batch sequence out_f -> batch sequence out_f")
        final_part = self.linear_2(mid_part)
        
        return final_part