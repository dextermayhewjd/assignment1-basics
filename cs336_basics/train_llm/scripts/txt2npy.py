import re
import numpy as np
from pathlib import Path
counts = []

ASSIGNMENT_REPO = Path("/home/fredkeira/projects/assignment1-basics")
TXT_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/chunk_tokens.txt"

NPY_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/chunk_token_counts.npy"

with open(TXT_PATH, "r") as f:
    for line in f:
        m = re.search(r":\s*([\d,]+)\s*tokens", line)
        if m:
            counts.append(int(m.group(1).replace(",", "")))

counts = np.array(counts, dtype=np.int64)
np.save(NPY_PATH, counts)
