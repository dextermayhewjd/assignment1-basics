import torch.nn as nn
import torch
import math

class Embedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 device: torch.device | None = None,
                 dtype: torch.dtype | None = None
                 ):
        super().__init__()
        '''
        um_embeddings: int 
                      Size of the vocabulary
        embedding_dim: int 
                      Dimension of the embedding vectors, 
                      i.e., dmodel
        device: torch.device | None = None 
                      Device to store the parameters on
        dtype: torch.dtype | None = None 
                      Data type of the parameters
        '''
        mean = 0
        std = math.sqrt(1)
        
        #用empty创建
        embedding = torch.empty(num_embeddings,
                                embedding_dim,
                                device= device,
                                dtype=dtype)
        
        torch.nn.init.trunc_normal_(
          tensor=embedding,
          mean= mean,
          std= std,
          a=-3*std,
          b=3*std
        )
        
        self.weights = nn.Parameter(data=embedding)
        
        
    def forward(self,token_ids:torch.Tensor)-> torch.Tensor:
        return self.weights[token_ids]