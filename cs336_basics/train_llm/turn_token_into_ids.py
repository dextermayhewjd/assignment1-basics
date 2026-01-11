from pathlib import Path
import numpy as np





# ============================================================
# Paths
# ============================================================

DATA_REPO = Path("/home/fredkeira/data")
ASSIGNMENT_REPO = Path("/home/fredkeira/projects/assignment1-basics")

TRAIN_DATA = DATA_REPO / "owt_train.txt"
VALID_DATA = DATA_REPO / "owt_valid.txt"

VOCAB_PATH = ASSIGNMENT_REPO / "bpe_outputs/owt_vocab.pkl"
MERGES_PATH = ASSIGNMENT_REPO / "bpe_outputs/owt_merges.pkl"

SPECIAL_TOKENS = ["<|endoftext|>"]

N = 2_727_120_452

tokens_mm = np.memmap(
    "owt_train_tokens.bin",
    dtype=np.int32,      # vocab_size < 2^31
    mode="w+",
    shape=(N,)
)