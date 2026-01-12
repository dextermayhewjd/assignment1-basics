import numpy as np
from pathlib import Path

ASSIGNMENT_REPO = Path("/home/fredkeira/projects/assignment1-basics")
NPY_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/chunk_token_counts.npy"

counts = np.load(NPY_PATH)

print(counts)

print(type(counts))         # 是什么对象
print(counts.shape)        # 形状
print(counts.dtype)        # 数据类型
print(len(counts))          # 元素数量

# offsets = np.zeros(len(counts) + 1, dtype=np.int64)

# prefix_sum = 0
# offsets[0] = 0
# for i, count in enumerate(counts):
#     prefix_sum += count
#     offsets[i + 1] = prefix_sum
# # print(offsets)

# start = offsets[:-1]
# end = offsets[1:]

# for i in range(len(counts)):
#     print(f"Chunk {i}: start={start[i]}, end={end[i]}, count={counts[i]}")