"""
train_bpe_tinystories.py

CS336 Assignment 1
Problem: train_bpe_tinystories

This script:
1. Trains a byte-level BPE tokenizer on TinyStories
2. Adds <|endoftext|> as a special token
3. Measures wall-clock time and memory usage
4. Saves vocab and merges to disk
5. Reports the longest token
6. Profiles runtime to identify bottlenecks
"""

import os
import time
import pickle
import psutil
import cProfile
import pstats
from pathlib import Path
from cs336_basics.debug.parallel_train_bpe_cache import train_bpe

# =========================
# CONFIG
# =========================
TINYSTORIES_PATH = "/home/dexterding/data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]

OUT_DIR = Path("/home/dexterding/projects/assignment1-basics/bpe_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_PATH = os.path.join(OUT_DIR, "tinystories_vocab.pkl")
MERGES_PATH = os.path.join(OUT_DIR, "tinystories_merges.pkl")
PROFILE_PATH = os.path.join(OUT_DIR, "profile.txt")

# =========================
# RESOURCE TRACKING SETUP
# =========================
process = psutil.Process(os.getpid())

def mem_gb():
    return process.memory_info().rss / 1024 ** 3


# =========================
# MAIN
# =========================
def main():
    print("=== Training BPE tokenizer on TinyStories ===")

    start_mem = mem_gb()
    start_time = time.time()

    profiler = cProfile.Profile()
    profiler.enable()

    vocab, merges = train_bpe(
        input_path=TINYSTORIES_PATH,
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
    )

    profiler.disable()

    end_time = time.time()
    end_mem = mem_gb()

    # =========================
    # SAVE ARTIFACTS
    # =========================
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)

    with open(MERGES_PATH, "wb") as f:
        pickle.dump(merges, f)

    # =========================
    # LONGEST TOKEN
    # =========================
    longest_token = max(vocab.values(), key=len)

    # =========================
    # PROFILING OUTPUT
    # =========================
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    with open(PROFILE_PATH, "w") as f:
        stats.stream = f
        stats.print_stats(30)

    # =========================
    # REPORT
    # =========================
    print("\n=== RESULTS ===")
    print(f"Training time: {(end_time - start_time) / 60:.2f} minutes")
    print(f"Approx. memory used: {end_mem - start_mem:.2f} GB")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Longest token length (bytes): {len(longest_token)}")
    print("Longest token (utf-8, errors=replace):")
    print(longest_token.decode('utf-8', errors='replace'))

    print("\nSaved files:")
    print(f"  vocab  -> {VOCAB_PATH}")
    print(f"  merges -> {MERGES_PATH}")
    print(f"  profile-> {PROFILE_PATH}")


if __name__ == "__main__":
    main()