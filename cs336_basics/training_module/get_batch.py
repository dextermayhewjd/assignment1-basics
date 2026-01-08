import numpy as np
import torch
import numpy.typing as npt
def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(dataset)
    max_start = n - context_length - 1
    assert max_start > 0 ,'max_start_point <0'
    
    # res1 = torch.empty(batch_size,context_length,dtype=torch.long,device=device)
    # res2 = torch.empty(batch_size,context_length,dtype=torch.long,device=device)
    
    # for i in range(batch_size):
    #   start_point = random.randint(0,max_start_point)
    #   res1[i] = torch.tensor(dataset[start_point:start_point+context_length],dtype=torch.long,device=device)
    #   res2[i] = torch.tensor(dataset[start_point+1:start_point+context_length+1],dtype=torch.long,device=device)
      
      
    # return res1,res2
    
    '''
    把上述的代码张量化 
    '''
    # 一次性随机生成 batch_size 个 start_point
    starts = torch.randint(0, max_start + 1, (batch_size,))
    
    #[0, 1, 2, 3, 4] “一个窗口内部的相对位置”
    offsets = torch.arange(context_length + 1)
    
    # 广播机制 一次性生成
    idx = starts[:, None] + offsets[None, :]
    
    batch = torch.from_numpy(dataset[idx.numpy()]).to(device)
    
    x = batch[:, :-1]
    y = batch[:, 1:]
    
    return x,y