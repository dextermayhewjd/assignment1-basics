from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import time
import sys

from cs336_basics.final_solutions.tokenizer2 import Tokenizer
from cs336_basics.pretokenization_example import find_chunk_boundaries


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


# ============================================================
# Global tokenizer (per worker)
# ============================================================

_GLOBAL_TOKENIZER: Tokenizer | None = None


def _init_worker(vocab_path: Path, merges_path: Path, special_tokens: list[str]):
    """
    每个 worker 进程启动时调用一次
    """
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )


# ============================================================
# Worker function (with heartbeat)
# ============================================================

def _count_tokens_in_chunk_worker(args):
    """
    真正跑在 worker 里的函数
    带 stderr heartbeat，用于证明 worker 仍在工作
    """
    file_path, start, end, chunk_id = args
    tokenizer = _GLOBAL_TOKENIZER
    assert tokenizer is not None

    pid = os.getpid()
    last_report = time.time()

    total = 0
    with open(file_path, "rb") as f:
        f.seek(start)

        # 如果不是 chunk 起点，丢掉残缺的一行
        if start != 0:
            f.readline()

        while f.tell() < end:
            line = f.readline()
            if not line:
                break

            text = line.decode("utf-8", errors="ignore")
            for _ in tokenizer.encode_iterable([text]):
                total += 1

            # ===== heartbeat：每 60 秒一次 =====
            now = time.time()
            if now - last_report > 60:
                print(
                    f"[worker pid={pid} | chunk={chunk_id}] alive "
                    f"offset={f.tell()} tokens≈{total}",
                    file=sys.stderr,
                    flush=True,
                )
                last_report = now

    return chunk_id, total


# ============================================================
# Parallel counting
# ============================================================

def parallel_count_tokens(
    file_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    num_processes: int 
) -> int:


    # -------- chunk boundaries --------
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_processes * 8,
            split_special_token=special_tokens[0].encode("utf-8"),
        )

    tasks = [
        (file_path, start, end, chunk_id)
        for chunk_id, (start, end) in enumerate(
            zip(boundaries[:-1], boundaries[1:])
        )
    ]

    total_tokens = 0
    per_chunk_tokens: dict[int, int] = {}

    with Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(vocab_path, merges_path, special_tokens),
    ) as pool:
        for chunk_id, cnt in tqdm(
            pool.imap_unordered(_count_tokens_in_chunk_worker, tasks),
            total=len(tasks),
            desc=f"Counting tokens in {file_path.name}",
            unit="chunk",
            dynamic_ncols=True,
        ):
            per_chunk_tokens[chunk_id] = cnt
            total_tokens += cnt

    # -------- optional: print per-chunk stats --------
    print(f"\nPer-chunk token counts for {file_path.name}:")
    for cid in sorted(per_chunk_tokens):
        print(f"  Chunk {cid:02d}: {per_chunk_tokens[cid]:,} tokens")

    return total_tokens


# ============================================================
# Main
# ============================================================

def main():
    num_workers = 20

    train_tokens = parallel_count_tokens(
        TRAIN_DATA,
        VOCAB_PATH,
        MERGES_PATH,
        SPECIAL_TOKENS,
        num_workers,
    )

    valid_tokens = parallel_count_tokens(
        VALID_DATA,
        VOCAB_PATH,
        MERGES_PATH,
        SPECIAL_TOKENS,
        num_workers,
    )

    print(f"\nTrain tokens: {train_tokens:,}")
    print(f"Valid tokens: {valid_tokens:,}")


if __name__ == "__main__":
    main()
