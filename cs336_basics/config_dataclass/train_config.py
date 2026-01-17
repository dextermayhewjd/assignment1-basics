from dataclasses import dataclass
from pathlib import Path
@dataclass
class TrainConfig:
    batch_size:int = 32
    max_steps:int = 100_000
    seed:int = 1337
    save_every:int = 500
    ckpt_dir:Path = Path("/home/fredkeira/data/cs336_training_result")
    