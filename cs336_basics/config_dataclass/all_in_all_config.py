from dataclasses import dataclass
from cs336_basics.config_dataclass.model_config import ModelConfig
from cs336_basics.config_dataclass.adamw_config import AdamWConfig
from cs336_basics.config_dataclass.lr_config import LRConfig

@dataclass
class Config:
    model: ModelConfig = ModelConfig()
    optim: AdamWConfig =AdamWConfig()
    lr_schedulr: LRConfig = LRConfig()