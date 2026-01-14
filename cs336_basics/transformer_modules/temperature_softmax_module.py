import torch
def temp_softmax(
            in_features:torch.Tensor,
            dimension:int,
            temp:float):
        """温度softmax函数
        Args:
                in_features (torch.Tensor): 输入张量
                
                dimension (int): 归一化的维度 
                一般来说是 [batch,seq_len,vocab_size]中的vocab_size维度
                所以 dimension= -1
                
                temp (float): 温度参数
        Returns:
                torch.Tensor: 归一化后的张量
        """
        in_dtype = in_features.dtype
        in_features = in_features.to(torch.float32)
        '''
        这里转换精度是为了防止下溢出
        因为exp函数对大负数的下溢非常敏感
        如果dtype是fp16或者bf16 
        '''
        
        # scale the logits by temperature
        in_features = in_features/temp
        # 为了数值稳定性 减去max
        max_x = torch.max(input= in_features,dim= dimension,keepdim=True).values
        # 注意这里的 max 返回的是除了values还有别的
        # 所以需要.values
        tensor_stable = in_features - max_x
        exp_tensor = torch.exp(tensor_stable)
        sum_exp = torch.sum(input=exp_tensor,dim=dimension,keepdim=True)
        res =  exp_tensor/sum_exp
        
        return res.to(in_dtype)
