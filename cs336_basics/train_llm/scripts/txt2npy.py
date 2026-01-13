import re
import numpy as np
from pathlib import Path
counts = []

ASSIGNMENT_REPO = Path("/home/fredkeira/projects/assignment1-basics")

OWT_TRAIN_TXT_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/scripts/owt_train_chunk_tokens.txt"
OWT_VALID_TXT_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/scripts/owt_valid_chunk_tokens.txt"

OWT_TRAIN_NPY_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/scripts/owt_train_chunk_token_counts.npy"
OWT_VALID_NPY_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/scripts/owt_valid_chunk_token_counts.npy"

# with open(OWT_TRAIN_TXT_PATH, "r") as f:
#     for line in f:
#         m = re.search(r":\s*([\d,]+)\s*tokens", line)
#         if m:
#             counts.append(int(m.group(1).replace(",", "")))

# counts = np.array(counts, dtype=np.int64)
# np.save(OWT_TRAIN_NPY_PATH, counts)


with open(OWT_VALID_TXT_PATH, "r") as f:
    for line in f:
        m = re.search(r":\s*([\d,]+)\s*tokens", line)
        if m:
            counts.append(int(m.group(1).replace(",", "")))

counts = np.array(counts, dtype=np.int64)
np.save(OWT_VALID_NPY_PATH, counts)