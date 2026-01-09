import torch
import os
from typing import BinaryIO, IO, Union
'''
every nn.Module has a state_dict() method that returns a dictionary with all learnable
weights;
we can restore these weights later with the sister method load_state_dict(). 
The same goes for any nn.optim.Optimizer. 
Finally, torch.save(obj,dest) can dump an object (e.g., adictionary
containing tensors in some values,but also regular Python objects like integers)
to a file(path) orfile-like object,
which can then be loaded back into memory with torch.load(src).
'''

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
  checkpoint = {}

  checkpoint["model"] = model.state_dict()
  checkpoint["optimizer"] = optimizer.state_dict()
  checkpoint["iteration"] = iteration

  
  torch.save(checkpoint, out)
  
def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    checkpoint = torch.load(src, weights_only=True) # 建议开启 weights_only 提高安全性
        
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # 直接从字典返回，不要再次 load 文件
    return checkpoint["iteration"]