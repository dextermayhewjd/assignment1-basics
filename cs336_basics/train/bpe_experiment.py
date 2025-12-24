import time
from cs336_basics.debug.parallel_train_bpe_cache import train_bpe

# === 改成你机器上的真实路径 ===
TINYSTORIES_PATH = "/data/tinystories.txt"
OWT_PATH = "/data/openwebtext.txt"

# ---------- TinyStories ----------
start = time.time()
vocab, merges = train_bpe(
    input_path=TINYSTORIES_PATH,
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
)
elapsed = time.time() - start

print("TinyStories")
print("Time (minutes):", elapsed / 60)
print("Vocab size:", len(vocab))
print("Longest token length:", max(len(v) for v in vocab.values()))