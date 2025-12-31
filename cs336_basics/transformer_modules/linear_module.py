import torch
import torch.nn as nn  
import math
from einops import einsum

class Linear(nn.Module):
    def __init__(self,
               in_features: int,
               out_features:int,
               device: torch.device |None=None,
               dtype: torch.dtype |None=None):
        super().__init__()
        
        W= torch.empty(out_features,
                       in_features,
                       device=device,
                       dtype=dtype)
        
        mean= 0
        std= math.sqrt(2/(in_features+out_features))
        
        torch.nn.init.trunc_normal_(
          tensor=W,
          mean=mean,
          std=std,
          a=-3*std,
          b=3*std
          )
        
        self.W = nn.Parameter(W)

        
    def forward(self,x:torch.Tensor)->torch.Tensor:
        # return x@self.W.T
        
        return einsum(x,self.W,'batch sequence in_f, out_f in_f -> batch sequence out_f')