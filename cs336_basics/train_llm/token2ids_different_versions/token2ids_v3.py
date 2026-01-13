"""
Docstring: Optimized Parallel Tokenizer (Direct Memmap Write)
- 移除了 Queue 和 Writer 进程
- 采用每个 Worker 独立写入 memmap 的方式 (Zero-Copy / OS Page Cache)
"""

from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm
import numpy as np

# 假设你的模块路径没变
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
# Globals (per worker process)
# ============================================================

_GLOBAL_TOKENIZER: Tokenizer | None = None

def _init_worker(vocab_path: Path, merges_path: Path, special_tokens: list[str]):
    """
    Worker 初始化：仅需加载 Tokenizer，不再需要 Queue
    """
    global _GLOBAL_TOKENIZER
    _GLOBAL_TOKENIZER = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )

# ============================================================
# Chunk boundary helpers
# ============================================================

def align_boundary_to_line_start(f, offset: int) -> int:
    if offset == 0:
        return 0
    pos = offset
    while pos > 0:
        f.seek(pos - 1)
        if f.read(1) == b"\n":
            return pos
        pos -= 1
    return 0

def build_line_aligned_boundaries(
    file_path: Path,
    special_token: bytes = b"<|endoftext|>",
    num_chunks: int = 160,
) -> list[int]:
    with open(file_path, "rb") as f:
        rough_boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_chunks,
            split_special_token=special_token,
        )
        line_aligned = [align_boundary_to_line_start(f, off) for off in rough_boundaries]
    return sorted(set(line_aligned))

# ============================================================
# Worker: Tokenize -> Direct Write to Memmap
# ============================================================

def _encode_tokens_direct_write(args):
    (
        chunk_id,
        file_path,
        start_offset,     # 文件字节读取起点
        end_offset,       # 文件字节读取终点
        token_start_idx,  # memmap 写入起点 (offsets[i])
        expected_count,   # 校验用
        out_bin_path,     # 输出文件路径
        dtype_str,        # 数据类型
        total_tokens,     # memmap 总大小 (用于 shape)
        flush_limit,      # 内存 buffer 大小
    ) = args

    tokenizer = _GLOBAL_TOKENIZER
    dtype = np.dtype(dtype_str)

    # 【关键优化】Worker 独立打开 memmap
    # mode="r+" 表示打开已存在文件进行读写。
    # 只要不同 worker 写入的区域不重叠，OS 会完美处理并发。
    mm = np.memmap(out_bin_path, dtype=dtype, mode="r+", shape=(total_tokens,))

    # 本地写入游标
    cursor = int(token_start_idx)
    emitted = 0
    buf = []

    with open(file_path, "rb") as f:
        f.seek(start_offset)
        
        while f.tell() < end_offset:
            line = f.readline()
            if not line:
                break
            
            text = line.decode("utf-8", errors="ignore")
            
            # Tokenize
            for tid in tokenizer.encode_iterable([text]):
                buf.append(tid)
                emitted += 1

                # 批量写入 memmap (减少 page fault 频率)
                if len(buf) >= flush_limit:
                    arr = np.asarray(buf, dtype=dtype)
                    write_end = cursor + len(arr)
                    mm[cursor : write_end] = arr # 直接赋值给磁盘映射区
                    cursor = write_end
                    buf.clear()

    # Flush 剩余部分
    if buf:
        arr = np.asarray(buf, dtype=dtype)
        write_end = cursor + len(arr)
        mm[cursor : write_end] = arr
        cursor = write_end
        buf.clear()

    # 简单的本地 flush，确保数据落盘（可选，OS 也会自动做）
    # mm.flush() 
    
    # 校验
    if emitted != expected_count:
        # 注意：这里如果报错，主进程会收到异常
        raise RuntimeError(
            f"[chunk {chunk_id}] Mismatch! Expected {expected_count}, got {emitted}"
        )
        
    return chunk_id, emitted

# ============================================================
# Orchestrator
# ============================================================

def parallel_encode_direct_write(
    file_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    out_bin_path: Path,
    dtype: np.dtype,
    total_tokens: int,
    num_processes: int,
    offsets: np.ndarray,
    counts: np.ndarray,
):
    # 1. 计算 Chunk 边界
    boundaries = build_line_aligned_boundaries(
        file_path=file_path,
        special_token=b"<|endoftext|>",
        num_chunks=160,
    )
    num_chunks = len(boundaries) - 1

    # 2. 【关键步骤】预创建文件
    # 使用 w+ 模式创建并设置好文件大小。
    # 这一步必须在单进程中完成，否则多进程 open w+ 会互相截断文件。
    print(f"Pre-allocating file: {out_bin_path} ({total_tokens} tokens)...")
    mm = np.memmap(out_bin_path, dtype=dtype, mode="w+", shape=(total_tokens,))
    # 这一步不是必须的，但对于某些文件系统，预填充可能有助于性能，
    # 不过为了速度通常只需要占位即可。
    del mm  # 关闭句柄，让 workers 重新打开
    
    dtype_str = np.dtype(dtype).str

    # 3. 准备任务
    tasks = []
    for i in range(num_chunks):
        tasks.append((
            i,
            file_path,
            boundaries[i],
            boundaries[i+1],
            int(offsets[i]),   # 该 chunk 在 memmap 中的起始索引
            int(counts[i]),
            out_bin_path,      # 传递路径，让 worker 自己打开
            dtype_str,
            total_tokens,
            500_000            # flush_limit
        ))

    # 4. 并行执行
    # 注意：不再需要 Queue，也不再需要 Writer 进程
    ctx = mp.get_context("fork") # 或 spawn
    with ctx.Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(vocab_path, merges_path, special_tokens),
    ) as pool:
        
        results = []
        for res in tqdm(
            pool.imap_unordered(_encode_tokens_direct_write, tasks),
            total=len(tasks),
            desc=f"Tokenizing {file_path.name}",
            smoothing=0.05
        ):
            results.append(res)
            
    # 5. 最终校验
    # 再次打开检查一下（可选）
    print("Verifying final file...")
    mm_check = np.memmap(out_bin_path, dtype=dtype, mode="r", shape=(total_tokens,))
    # 这里只能做简单的长度/非空校验，或者你可以根据 logic 做抽样检查
    # 由于每个 worker 都做了 emitted == expected_count 校验，这里其实比较安全
    print(f"Done. File saved to {out_bin_path}")


# ============================================================
# main
# ============================================================

def main():
    num_workers = 20      
    
    # 路径配置
    # 使用绝对路径

    
    
    
    
    OUTPUT_DATA_REPO = ASSIGNMENT_REPO / "token_to_id_outputs"
    OUTPUT_DATA_REPO.mkdir(parents=True, exist_ok=True)

    # OWT_VALID_NPY_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/scripts/owt_valid_chunk_token_counts.npy"
    # valid_token_arr_path = OUTPUT_DATA_REPO / "valid_ids.bin"
    # N_valid_num = 66_401_048
    
    # # 加载 counts 并计算 offsets
    # counts = np.load(OWT_VALID_NPY_PATH)
    # offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    # prefix_sum = 0
    # offsets[0] = 0
    # for i, count in enumerate(counts):
    #     prefix_sum += count
    #     offsets[i + 1] = prefix_sum
        
    # if int(offsets[-1]) != N_valid_num:
    #     raise RuntimeError(f"Count mismatch: {offsets[-1]} != {N_valid_num}")

    # # 执行并行编码
    # parallel_encode_direct_write(
    #     file_path=VALID_DATA,
    #     vocab_path=VOCAB_PATH,
    #     merges_path=MERGES_PATH,
    #     special_tokens=SPECIAL_TOKENS,
    #     out_bin_path=valid_token_arr_path,
    #     dtype=np.uint16,
    #     total_tokens=N_valid_num,
    #     num_processes=num_workers,
    #     offsets=offsets,
    #     counts=counts,
    # )
    
    OWT_TRAIN_NPY_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/scripts/owt_train_chunk_token_counts.npy"
    train_token_arr_path = OUTPUT_DATA_REPO / "train_ids.bin"
    N_train_num = 2_727_120_424
    
    # 加载 counts 并计算 offsets
    counts = np.load(OWT_TRAIN_NPY_PATH)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)
    prefix_sum = 0
    offsets[0] = 0
    for i, count in enumerate(counts):
        prefix_sum += count
        offsets[i + 1] = prefix_sum
        
    if int(offsets[-1]) != N_train_num:
        raise RuntimeError(f"Count mismatch: {offsets[-1]} != {N_train_num}")

    # 执行并行编码
    parallel_encode_direct_write(
        file_path=TRAIN_DATA,
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=SPECIAL_TOKENS,
        out_bin_path=train_token_arr_path,
        dtype=np.uint16,
        total_tokens=N_train_num,
        num_processes=num_workers,
        offsets=offsets,
        counts=counts,
    )


if __name__ == "__main__":
    main()