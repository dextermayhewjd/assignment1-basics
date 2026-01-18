from omegaconf import OmegaConf
from cs336_basics.config.all_in_all_config import Config
from pathlib import Path

config_path = Path("/home/fredkeira/projects/assignment1-basics/cs336_basics/config/config.yaml")

base_cfg = OmegaConf.structured(Config)
file_cfg = OmegaConf.load(config_path)
cfg = OmegaConf.merge(base_cfg, file_cfg)

type(cfg)
print(cfg)