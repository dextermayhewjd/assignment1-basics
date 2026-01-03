import torch
import torch.nn as nn 
from cs336_basics.transformer_modules.multihead_self_attention import Multihead_Self_Attention
from cs336_basics.transformer_modules.rope_module import RoPE
from einops import einsum,rearrange
from cs336_basics.transformer_modules.scaled_dot_product_attention import scale_dot_product_attention

class MultiheadSelfAttentionWithRope(Multihead_Self_Attention):
    def __init__(self,
                 d_model:int,
                 num_heads:int,
                 theta:float,
                 max_seq_len:int):
      
        super().__init__(d_model, num_heads)
        self.rope = RoPE(theta= theta,
                         max_seq_len=max_seq_len,
                         d_k= d_model//num_heads
                         )
    def forward(self,x,token_positions):
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

        q_lower_with_r = self.rope.forward(x=q_lower,token_positions=token_positions)
        k_lower_with_r = self.rope.forward(x=k_lower,token_positions=token_positions)
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
        
        out_put = scale_dot_product_attention(Q=q_lower_with_r,K=k_lower_with_r,V=v_lower,mask=mask)
        
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