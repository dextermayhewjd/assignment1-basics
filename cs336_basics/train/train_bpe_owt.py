import os
import time
import pickle
import psutil
import argparse
import cProfile
import pstats
import threading
from pathlib import Path

from cs336_basics.final_solutions.parallel_train_bpe_cache import train_bpe

# =========================
# CONFIG
# =========================
TINYSTORIES_PATH = "/home/dexterding/data/owt_train.txt"
VOCAB_SIZE = 32_000
SPECIAL_TOKENS = ["<|endoftext|>"]

OUT_DIR = Path("/home/dexterding/projects/assignment1-basics/bpe_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_PATH = OUT_DIR / "owt_vocab.pkl"
MERGES_PATH = OUT_DIR / "owt_merges.pkl"
PROFILE_PATH = OUT_DIR / "profile_owt.txt"


# =========================
# MEMORY MONITOR (Peak, main + children)
# =========================
_PARENT = psutil.Process(os.getpid())


def _rss_bytes_process_tree() -> int:
    """Sum RSS of current process + all children (recursive)."""
    total = 0
    procs = [_PARENT] + _PARENT.children(recursive=True)
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return total


def _gb(x_bytes: int) -> float:
    return x_bytes / (1024 ** 3)


class PeakMemMonitor:
    """Poll RSS (main+children) and record peak."""
    def __init__(self, interval_sec: float = 0.2):
        self.interval_sec = interval_sec
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            cur = _rss_bytes_process_tree()
            if cur > self.peak_bytes:
                self.peak_bytes = cur
            time.sleep(self.interval_sec)

    def __enter__(self):
        # 初始化一下，避免 peak=0
        self.peak_bytes = _rss_bytes_process_tree()
        self._t.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        self._t.join()


# =========================
# Runner
# =========================
def run_train(profile: bool):
    start_time = time.time()

    # 原来的“主进程净增RSS”（保留做对比）
    start_rss_main = _PARENT.memory_info().rss

    with PeakMemMonitor(interval_sec=0.2) as mon:
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()
            vocab, merges = train_bpe(
                input_path=TINYSTORIES_PATH,
                vocab_size=VOCAB_SIZE,
                special_tokens=SPECIAL_TOKENS,
            )
            profiler.disable()

            prof_path = PROFILE_PATH.with_suffix(".prof")
            profiler.dump_stats(prof_path)   # ✅ 保存 .prof

            with open(PROFILE_PATH, "w") as f:
                stats = pstats.Stats(profiler, stream=f).sort_stats("cumtime")
                stats.print_stats(30)
        else:
            vocab, merges = train_bpe(
                input_path=TINYSTORIES_PATH,
                vocab_size=VOCAB_SIZE,
                special_tokens=SPECIAL_TOKENS,
            )

    end_time = time.time()
    end_rss_main = _PARENT.memory_info().rss

    # save
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    with open(MERGES_PATH, "wb") as f:
        pickle.dump(merges, f)

    # longest token (after training; cheap)
    longest_token = max(vocab.values(), key=len)

    print("\n=== RESULTS ===")
    print(f"Training time (this run): {(end_time - start_time)/60:.2f} minutes")

    # 峰值（主进程+子进程）
    print(f"Peak RSS (main + children): {_gb(mon.peak_bytes):.2f} GB")

    # 可选：保留原来那种“净增”（通常会小得离谱，用来对比就好）
    print(f"Main RSS delta (end-start): {_gb(end_rss_main - start_rss_main):.2f} GB")

    print(f"Vocab size: {len(vocab)} | merges: {len(merges)}")
    print(f"Longest token bytes: {len(longest_token)}")
    print(longest_token.decode('utf-8', errors='replace'))

    print("\nSaved:")
    print(" ", str(VOCAB_PATH))
    print(" ", str(MERGES_PATH))
    if profile:
        print(" ", str(PROFILE_PATH))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true", help="Enable cProfile (will slow down)")
    args = ap.parse_args()
    run_train(profile=args.profile)


if __name__ == "__main__":
    main()
