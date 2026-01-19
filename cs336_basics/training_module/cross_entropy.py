from torch import Tensor
from jaxtyping import Float,Int
import torch
from einops import rearrange,reduce

def cross_entropy(
    logits: Float[Tensor, "batch time vocab_size"],
    targets: Int[Tensor, "batch time"],
) -> Float[Tensor, ""]:
        
    # 这个不是 doc string 只是注释
    # 这里是原来的理解 但是可以向量化加速
    '''    
    # 找到max的logits 用来避免上溢风险 
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
from torch import Tensor
import torch
from jaxtyping import Float, Int

def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    targets: Int[Tensor, "..."],
) -> Float[Tensor, ""]:
    """
    A numerically stable cross entropy loss.

    Supports:
      - logits: (N, V), targets: (N,)
      - logits: (B, T, V), targets: (B, T)
    """

    # -------- 1. reshape to (N, V) --------
    V = logits.shape[-1]
    logits = logits.reshape(-1, V)
    targets = targets.reshape(-1)

    # -------- 2. log-sum-exp (stable) --------
    logits_max = logits.max(dim=-1, keepdim=True).values
    logsumexp = torch.log(
        torch.exp(logits - logits_max).sum(dim=-1)
    ) + logits_max.squeeze(-1)

    # -------- 3. gather correct class --------
    row_idx = torch.arange(logits.shape[0], device=logits.device)
    selected = logits[row_idx, targets]

    # -------- 4. cross entropy --------
    loss = -selected + logsumexp
    return loss.mean()
