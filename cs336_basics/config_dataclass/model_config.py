from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layer: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: int = 10_000