import torch
from torch import Tensor
import torch.nn as nn 
from jaxtyping import Int

from cs336_basics.transformer_modules.embedding_module import Embedding
from cs336_basics.transformer_assembling.transformer_block import Transformer_Block
from cs336_basics.transformer_modules.linear_module import Linear
from cs336_basics.transformer_modules.rmsnorm_module import RMSNorm
from cs336_basics.transformer_modules.softmax_module import softmax
 

class Transformer_LM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float
                 ):
        super().__init__()
        
        self.embedding = Embedding(num_embeddings=vocab_size,
                                   embedding_dim=d_model)
        self.blocks = nn.ModuleList([
            Transformer_Block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=rope_theta,
            )
            for _ in range(num_layers)
        ]) ## list comprehension 

        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)      
        
    def forward(self, in_indices,  num_layers: int | None = None):
      """
      num_layers:
          - None: 用全部层
          - int: 只跑前 num_layers 层
      """
      x = self.embedding(in_indices)

      L = num_layers if num_layers is not None else len(self.blocks)

      for block in self.blocks[:L]:
          x = block(x)

      x = self.norm(x)
      logits = self.lm_head(x)
      return logits
      return softmax(in_features=logits,dimension= -1)