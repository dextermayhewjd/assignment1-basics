import torch
from cs336_basics.transformer_modules.temperature_softmax_module  import temp_softmax
def top_p_sample_next_token(
            input_features:torch.Tensor,
            dimension:int,
            temp:float,
            p:float
            
            ):
        """Top-p (nucleus) sampling 函数
        args:
                input_features (torch.Tensor): 输入张量
                dimension (int): 归一化的维度 
                        一般来说是 [batch,seq_len,vocab_size]中的vocab_size维度
                        所以 dimension= -1
                temp (float): 温度参数 大于1会使分布更平坦 小于1会使分布更陡峭
                p (float): top-p 参数 范围在(0,1)
        returns:
        """
        
        # 1. 先用温度softmax计算概率分布
        scaled_probs = temp_softmax(
            in_features=input_features,
            dimension=dimension,
            temp=temp
        )
        # 2. 对概率分布进行top-p采样
        # 先对概率进行降序排序
        sorted_probs, sorted_indices = torch.sort(
                input=scaled_probs,
                dim=dimension,
                descending=True
        )
        # torch 的 prefix sum函数是 torch.cumsum
        cumulative_probs = torch.cumsum(
                input=sorted_probs,
                dim=dimension
        )
        ''' 
        下面是错误写法 高级index不是这么用的 
        
        # 找到截断位置
        # cumulative_probs > p 的第一个位置
        # 通过在cumulative_probs中找到第一个大于p的位置
        # 然后将该位置及其之前的所有位置都保留
        
               cutoff_mask = cumulative_probs > p
        # 第一个大于p的位置是true 之后的位置都是真
        # 获取第一个true的位置
        # torch.argmax 返回的是最大值第一次出现的index
        first_cutoff_index = torch.argmax(cutoff_mask.to(torch.int32),dim=dimension,keepdim=True)
        #修改使用原来的cutoff_mask
        cutoff_mask[first_cutoff_index] = False
        # 为了一会取反
        # 
        # '''
        
        # 找到截断位置 然后创建mask 但是要包含阶段的位置
        keep_mask = cumulative_probs < p
        # 包含第一个大于p的位置 通过逻辑或操作实现 正好其他地方都是0
        keep_mask = keep_mask |torch.roll(
                input= keep_mask,
                shifts=1,
                dims=dimension
        )
        '''
        keep_mask            = [ True,  True, False, False ]
        roll(keep_mask, 1)   = [ False, True,  True, False ]
        '''
        
        '''
        这里错了 因为batchsize>1的时候 不能简单地把第一个位置设为True
        应该是对每个batch单独处理
        # 确保至少保留一个token
        if keep_mask[0] == False:
            keep_mask[0] = True
        '''
        
            # Ensure at least one token is kept (per sample)
        index0 = [slice(None)] * keep_mask.dim()
        index0[dimension] = 0
        keep_mask[tuple(index0)] = True
        

        # 将cutoff_mask取反后应用到scaled_probs上
        masked_sorted_probs = torch.where(
                keep_mask,
                sorted_probs,
                torch.zeros_like(sorted_probs)
                
        )

        # 重归一化 
        masked_sum = torch.sum(
                input=masked_sorted_probs,
                dim=dimension,
                keepdim=True
        )
        normalized_probs = masked_sorted_probs / (masked_sum + 1e-8)
        
        #将值scatter 回原来的位置
        # 创建一个全0张量
        res_probs = torch.zeros_like(scaled_probs)
        res_probs.scatter_(
                dim=dimension,
                index=sorted_indices,
                src=normalized_probs
        )
        
        # 根据新的概率分布采样 返回这个token的index 也就是第几个
        next_token_id = torch.multinomial(
                input=res_probs,
                num_samples=1
        )
        return next_token_id