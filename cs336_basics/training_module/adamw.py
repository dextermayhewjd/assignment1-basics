
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params,         
                lr,
                weight_decay,
                betas,
                eps 
                ):
        
        # ===== defaults：param_group 的“兜底超参数”=====
        defaults = {"lr":lr,
                    "betas": betas,
                    "eps": eps,
                    "weight_decay": weight_decay
                    }
        
        # ===== 调用父类 Optimizer.__init__ =====
        # 这里会发生三件非常重要的事情：
        #
        # 1. self.param_groups 被构造：
        #    - 如果 params 是 Parameter iterable -> 单一 param_group
        #    - 如果 params 是 dict list -> 多 param_group
        #
        # 2. self.state 被创建：
        #    self.state = defaultdict(dict)
        #
        # 3. 但！！此时 self.state 里还没有任何 parameter 的状态
        #    （采用 lazy initialization）
        super().__init__(params, defaults)
        
    def step(self):
        # 有很多个group  每个group里 有两个一般 一个是 lr 一个是 parameters 例如nn.parameters
        for group in self.param_groups:
            
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                
                # 当前参数的梯度（来自 backward）
                grad = p.grad.data

                # ===== 取出该参数对应的 state =====
                # self.state 是 Optimizer 父类中定义的 defaultdict(dict)
                # 因此：
                # - 如果 p 第一次出现：self.state[p] == {}
                # - 如果 p 之前更新过：self.state[p] 是已有的状态字典
                state = self.state[p]
                
                if len(state) ==0:
                    state["step"] = 0
                
                    state["m"] = torch.zeros_like(p)
                    
                    state["v"] = torch.zeros_like(p)
                    
                m = state["m"]
                v = state["v"]
                
                # 更新 step 计数（per-parameter）
                state["step"] += 1
                t = state["step"]
                
                # ---------- (1) 一阶动量 ----------
                # m_t = β₁ m_{t-1} + (1 - β₁) g_t
                
                # m_new = (beta1*m) + (1-beta1)*grad
                m.mul_(beta1).add_(grad,alpha = 1 - beta1)
                
                # ---------- (2) 二阶动量 ----------
                # v_t = β₂ v_{t-1} + (1 - β₂) g_t²
                
                # v_new = beta2*v +(1-beta2)*grad*grad
                v.mul_(beta2).addcmul_(grad,grad,value = 1-beta2)
                
                # ---------- (3) Bias correction ----------
                # α_t = α · sqrt(1 - β₂^t) / (1 - β₁^t)
                bc2 = 1 - beta2**t
                bc1 = 1 - beta1**t
                alpha_t = lr * (math.sqrt(bc2)/bc1)
                
                
                # ---------- (4) Adam 更新 ----------
                # p.data = p.data - alpha_t*(m_new/(math.sqrt(v_new)+eps))
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t)
                
                # p.data = p.data - lr * weight_decay * grad_new
                if weight_decay != 0.0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)