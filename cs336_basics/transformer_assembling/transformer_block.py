import torch
import torch.nn as nn 

from cs336_basics.transformer_modules.multihead_self_attention_with_rope import MultiheadSelfAttentionWithRope

from cs336_basics.transformer_modules.positionwise_feedforward_moudule import SwiGLU_feed_forward
from cs336_basics.transformer_modules.rmsnorm_module import RMSNorm

from jaxtyping import Float
from torch import Tensor

class Transformer_Block(nn.Module):
    def __init__(self,
                  d_model: int,
                  num_heads: int,
                  d_ff: int,
                  max_seq_len: int,
                  theta: float
                 ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        self.mha_r = MultiheadSelfAttentionWithRope(
          d_model=self.d_model,
          num_heads= num_heads,
          theta= theta,
          max_seq_len= max_seq_len
        )
        
        self.swiglu_ff = SwiGLU_feed_forward(
          d_model=self.d_model,
          d_ff= self.d_ff
        )
        
        self.rmsnorm_1 = RMSNorm(d_model=d_model)
        self.rmsnorm_2 = RMSNorm(d_model=d_model)

        
    def forward(self,
                x: Float[Tensor, "batch sequence_length d_model"]):
        
        B,T,D = x.shape
        token_positions = torch.arange(T).unsqueeze(0)# [0,1,2,3...T-1] 后变成[[1,2,3,4,...T-1]]
        x_norm = self.rmsnorm_1(x)
        x_cmha_r = self.mha_r(x_norm,token_positions)
        
        x_add = x + x_cmha_r
        
        x_add_norm = self.rmsnorm_2(x_add)
        x_add_norm_ff = self.swiglu_ff(x_add_norm)
        
        x_out = x_add_norm_ff + x_add
        
        return x_out
      
        '''
        工程问题是 记得 两个模块的参数不共享 所以得有两个rmsnorm
        torch.arrange(T)不是很会用
        
        torch.arange(T)[None, :]
        [None,:] 是会创造处一个新的维度
        等价于
        torch.arange(T).unsqueeze(0)
        整理
        
        https://chatgpt.com/g/g-p-693f75d2365c8191baf9aaa7038e3595-cs336xiao-xi-jie/c/6958afa5-4ff0-832e-9ab1-def8fdf6c9c7
        
        世界是为什么 cos_cache 里可以使用tensor 以及只是最后一个维度
        '''
        
        