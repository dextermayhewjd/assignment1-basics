import torch
from collections.abc import Iterable

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_l2_norm: float):
    eps = 1e-6
    total_norm_sq = 0.0

    # 1. 计算整体 L2 norm 的平方
    for param in params:
        if param.grad is None:
            continue
        grad = param.grad.data
        total_norm_sq += grad.pow(2).sum()

    total_norm = torch.sqrt(total_norm_sq + eps)

    # 2. 如果不需要 clip，直接返回
    if total_norm <= max_l2_norm:
        return

    # 3. 计算缩放系数
    clip_coef = max_l2_norm / (total_norm + eps)

    # 4. 原地缩放所有梯度
    for param in params:
        if param.grad is None:
            continue
        param.grad.data.mul_(clip_coef)