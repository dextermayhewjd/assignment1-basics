from dataclasses import dataclass

@dataclass
class AdamWConfig:
    beta1: float = 0.9
    beta2: float = 0.95
    eps:float = 1e-8
    weight_decay: float = 0.1