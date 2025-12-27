import pickle

VOCAB_PATH = "/home/dexterding/projects/assignment1-basics/bpe_outputs/owt_vocab.pkl"  # 改成你的路径

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

# 1) 找到最长 token
tok_id, tok_bytes = max(vocab.items(), key=lambda kv: len(kv[1]))
print("longest_token_id:", tok_id)
print("byte_len:", len(tok_bytes))
print("raw_bytes_repr:", repr(tok_bytes))
print("decoded_utf8 (replace):", tok_bytes.decode("utf-8", errors="replace"))

# 2)（可选）看前 20 个最长的，方便判断“是否合理”
topk = 20
longest = sorted(vocab.items(), key=lambda kv: len(kv[1]), reverse=True)[:topk]
for i, (tid, tb) in enumerate(longest, 1):
    print(f"{i:02d} id={tid:>6} len={len(tb):>4} bytes={repr(tb)} text={tb.decode('utf-8', errors='replace')}")
