import torch
import torch.nn as nn 
import math 
class RMSNorm(nn.Module):
    '''
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
    '''
    def __init__(self,
              d_model: int,
              eps: float = 1e-5,
              device=None,
              dtype=None):
  
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        
        self.weights = nn.Parameter(
          torch.ones(d_model,device=device,dtype=dtype)
        )
        
        
    # input tensor 是（Batch_size,seq_len,d_model）
    def forward(self, x: torch.Tensor)-> torch.Tensor: 
        
        in_dtype = x.dtype
        x = x.to(torch.float32)
        
       # rms = math.sqrt(x.pow(2).mean(dim=-1,keepdim=True) + self.eps)
        '''
        pow(2) 是每个元素平方 mean (dim=-1) 是求平均 除以的是d_model
        等价于下面的公式 我觉得还是下面的比较好识别 
        同时记住要keepdim = True 
        '''
        rms = torch.sqrt(x.pow(2).sum(dim=-1,keepdim= True)/self.d_model + self.eps )
        '''
        !!! math.sqrt 不支持sqrt batch tensor 得需要用torch。sqrt
        '''
        x = x/rms
        x = x*self.weights
        
        return x.to(in_dtype)