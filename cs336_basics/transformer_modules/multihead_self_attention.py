import torch
import torch.nn as nn 

from jaxtyping import Float
from torch import Tensor
import math
from einops import einsum,rearrange
from cs336_basics.transformer_modules.softmax_module import softmax
from cs336_basics.transformer_modules.scaled_dot_product_attention import scale_dot_product_attention
class Multihead_Self_Attention(nn.Module):
    def __init__(self,
                 d_model:int,
                 num_heads:int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        w_Q = torch.empty(num_heads * self.d_head, d_model)
        w_K = torch.empty(num_heads * self.d_head, d_model)
        w_V = torch.empty(num_heads * self.d_head, d_model)
        
        self.w_Q = nn.Parameter(w_Q)
        self.w_K = nn.Parameter(w_K)
        self.w_V = nn.Parameter(w_V)
        
        self.w_O = nn.Parameter(torch.empty(d_model, d_model))
    def forward(self,
                x:Float[Tensor, "batch seq_len d_model"]
                )-> Float[Tensor, "batch seq_len d_model"]:
        
        B, T, _ = x.shape
        H = self.num_heads
        D = self.d_head
        
        '''
        下面两个都对
        但是要小心 ... 的时候 后面不能漏 不然会会报错
        '''        
        

        # q = x @ self.w_Q.T   # (B, T, H*D)
        # k = x @ self.w_K.T
        # v = x @ self.w_V.T

        q = einsum(x,self.w_Q,"... seq_len d_m, d_m2 d_m ->... seq_len d_m2")
        k = einsum(x,self.w_K,"... seq_len d_m, d_m2 d_m ->... seq_len d_m2")
        v = einsum(x,self.w_V,"... seq_len d_m, d_m2 d_m ->... seq_len d_m2")

        
        '''
        下面两个都对 
        '''
        # q_lower = q.view(B,T,H,D).transpose(1,2)
        # k_lower = k.view(B,T,H,D).transpose(1,2)
        # v_lower = v.view(B,T,H,D).transpose(1,2)
        
        q_lower = rearrange(q,"b t (h d) -> b h t d ",h= H)
        k_lower = rearrange(k,"b t (h d) -> b h t d ",h= H)
        v_lower = rearrange(v,"b t (h d) -> b h t d ",h= H)

        '''
        直接实现版本 
        '''
        # k_lower_T = k_lower.transpose(-2,-1)
        
        # ql_k_t= q_lower@k_lower_T
        # #ql_k_t = einsum(q_lower,k_lower,"b h seq_len d_head,b h seq_len d_head ->b h seq_len seq_len")
       
        # scaled_ql_k_t = ql_k_t/math.sqrt(D)
        
        # # === causal mask ===
        # mask = torch.tril(
        #     torch.ones(T, T,  dtype=torch.bool)
        # )
        # attn_scores = scaled_ql_k_t.masked_fill(~mask, float("-inf"))
        # ======================
        
        
              
        # attn_probs = softmax(in_features=attn_scores,dimension= -1)
  
        # out_put = attn_probs @ v_lower
        
        mask = torch.tril(
            torch.ones(T, T,device=x.device, dtype=torch.bool)
        )
        
        out_put = scale_dot_product_attention(Q=q_lower,K=k_lower,V=v_lower,mask=mask)
        
        # d_model = H*D
        # out_put_normal = out_put.transpose(1,2).contiguous().view(B,T,d_model)
        out_put_normal = rearrange(out_put,"b h seq_len d_head -> b seq_len (h d_head)")
        out = out_put_normal @ self.w_O.T
        return out
        
        '''
        明天实现rope 加上 把view 和 transpose和 contiguous()看一下 
        以及这里的 torch.tril比较简单 但是有无更好的更通用的解法呢
        masked_fill更加像是 true 就应用后面的float("-inf")
        '''