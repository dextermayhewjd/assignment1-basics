import torch
def softmax(in_features:torch.Tensor,
            dimension:int):
    
    max_x = torch.max(input=in_features,dim= dimension,keepdim=True).values
    # 注意这里的 max 返回的是除了values还有别的
    # 所以需要.values
    tensor_stable = in_features - max_x
    
    exp_tensor = torch.exp(tensor_stable)
    
    sum_exp = torch.sum(input=exp_tensor,dim=dimension,keepdim=True)

    return exp_tensor/sum_exp