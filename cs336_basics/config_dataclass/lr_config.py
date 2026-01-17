from dataclasses import dataclass

@dataclass
class LRConfig:
    lr_max: float = 1e-4
    lr_min: float = 1e-5
    warmup_step:int = 30_000
    cosine_step:int = 100_000
    