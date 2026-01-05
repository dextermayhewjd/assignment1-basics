from torch import Tensor
from jaxtyping import Float,Int
import torch
from einops import rearrange,reduce

def cross_entropy(
    logits: Float[Tensor,"batch vocab_size"],
    targets: Int[Tensor,"batch"]
    ) -> Float[Tensor, ""]:
        
    '''    # 找到max的logits 用来避免上溢风险 
        # logits_max = rearrange(logits,"... vocab_size -> ... 1 ","max")
        logits_max = reduce(logits,"... vocab_size -> ... 1 ","max")
        logits_max = torch.max(input=logits,dim=-1,keepdim=True).values
        # 记得这里必须使用max 
        
        # 为了消除指数爆炸 和 softmax中的一样
        safe_logits = logits-logits_max
        
        # softmax 公式 计算P
        exp_safe_logits = torch.exp(input=safe_logits)
        
        exp_sum = torch.sum(input=exp_safe_logits,dim=-1,keepdim= True)
        exp_sum = reduce(exp_safe_logits,"... vocab_size -> ... 1 ","sum")
        
        total_pb = exp_safe_logits/exp_sum
        # print(f"logits的尺寸{logits.shape[0]}")
        row_idx = torch.arange(logits.shape[0])          # (B,)
        col_idx = targets                            # (B,)

        selected_probability = total_pb[row_idx, col_idx]         # (B,)
        
        neg =  -torch.log(input=selected_probability)
        
        return torch.mean(neg)

    这里失效的原因是这个softmax的技巧解决的是上溢风险 
    一旦log 极小值 然后 -log 就变成极大值 就会上溢
    '''  
    logits_max = torch.max(input=logits,dim=-1,keepdim=True).values
    safe_logits = logits-logits_max
    exp_safe_logits = torch.exp(input=safe_logits)
    exp_sum = torch.sum(input=exp_safe_logits,dim=-1,keepdim= True)
    
    row_idx = torch.arange(logits.shape[0])          # (B,)
    col_idx = targets                            # (B,)

    selected_logit = safe_logits[row_idx, col_idx]         # (B,)
    
    # -(y-max) + log sum e^k-max
    out = - selected_logit + torch.log(input= exp_sum)
    return torch.mean(out)