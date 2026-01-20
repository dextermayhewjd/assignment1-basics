from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal

from enum import Enum

class MemmapDType(str, Enum):
    uint16 = "uint16"
    uint32 = "uint32"

class TorchDType(str, Enum):
    float32 = "float32"
    float16 = "float16"
    bfloat16 = "bfloat16"

# --- 1) Data ---
@dataclass
class DataConfig:
    # 存 train_ids.bin / valid_ids.bin 的目录
    data_dir: Path = "/home/fredkeira/projects/assignment1-basics/token_to_id_outputs"
    train_bin_name: str = "train_ids.bin"
    valid_bin_name: str = "valid_ids.bin"
    eot_id: int = 256
    # memmap dtype（现在用 uint16；如果以后 tokenizer vocab > 65535，就要 uint32）
    memmap_dtype: MemmapDType = MemmapDType.uint16
    
    
# --- 2) Model ---
@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: int = 10_000


# --- 3) Optim / Lr_Scheduler ---
@dataclass
class OptimConfig:
    # 这里是learning rate scheduler
    lr_max: float = 1e-4
    lr_min: float = 1e-5
    warmup_steps: int = 30_000
    cosine_steps: int = 100_000

    # adamw的参数
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    weight_decay: float = 0.1



# --- 4) Train ---
@dataclass
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 100_000
    seed: int = 1337

    max_l2_norm: float = 1.0  # grad clip

    save_every: int = 500
    eval_every: int = 100
    eval_steps: int = 50



# --- 5) Runtime ---
@dataclass
class RuntimeConfig:
    device: str = "cuda:1"
    dtype: TorchDType = TorchDType.float32



# --- 6) Experiment ---
@dataclass
class ExperimentConfig:
    run_dir: Path = "/home/fredkeira/data/cs336_training_result"
    exp_name: str = "tinystories_base"

    resume: bool = False
    resume_path: Optional[Path] = None
    
    debug_mode:bool = False
    debug_overfit: bool = False
    one_step: bool = False
    debug_shapes: bool = False
    debug_norms: bool = False
    # 如果指定，从这个 ckpt 恢复，否则默认用 ckpt_latest.pt


# --- Top-level Config ---

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)