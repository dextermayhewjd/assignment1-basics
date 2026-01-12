from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import time
import sys
import numpy as np

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

def align_boundary_to_line_start(f, offset: int) -> int:
    """
    将 byte offset（指向 <|endoftext|>）对齐到其所在行的行首
    """
    if offset == 0:
        return 0

    pos = offset
    while pos > 0:
        f.seek(pos - 1)
        if f.read(1) == b"\n":
            return pos
        pos -= 1

    return 0

# ============================================================

def build_line_aligned_boundaries(
    file_path: Path,
    special_token: bytes = b"<|endoftext|>",
    num_chunks: int = 160, # chunks数量目前是固定的和之前一样
) -> list[int]:
    """
    1. 用 special token 找粗 boundary
    2. 对齐到行首
    3. 打印所有 line-aligned boundary 的 byte offset
    """
    with open(file_path, "rb") as f:
        rough_boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_chunks,
            split_special_token=special_token,
        )
        line_aligned_boundaries = [
            align_boundary_to_line_start(f, offset)
            for offset in rough_boundaries
        ]
        
    aligned = sorted(set(line_aligned_boundaries))
    return aligned

# ============================================================
# encode tokens parallelly
# ============================================================

def _encode_tokens_in_chunk_worker(args):
    """
    每个 worker 进程调用此函数对指定 chunk 进行 tokenization
    """
    (
        chunk_id,
        file_path,
        start_offset,
        end_offset,
        token_id_arr,
        token_id_arr_start_idx,
        expected_count
    ) = args
    
    tokenizer = _GLOBAL_TOKENIZER
    idx = token_id_arr_start_idx
    
    with open(file_path, "rb") as f:
        f.seek(start_offset)
        
        
        while f.tell() < end_offset:
            line = f.readline()
            if not line:
                break
            
            text = line.decode("utf-8", errors="ignore")
            # tokenizer的encode_iterable方法会返回一个生成器
            # 使用生产器来给token_id_arr从token_id_arr_start_idx开始赋值
            
            for token_id in tokenizer.encode_iterable([text]):
                token_id_arr[idx] = token_id
                idx += 1
                
    written = idx - token_id_arr_start_idx
    if written != expected_count:
        raise RuntimeError(
            f"[chunk {chunk_id}] token mismatch: "
            f"expected {expected_count}, got {written}"
        )
    return chunk_id, written
# ============================================================

def parallel_encode_file_to_token_ids(
    file_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    token_id_arr: np.memmap,
    num_processes: int,
    offsets: np.ndarray,
    counts: np.ndarray,
    ):
    """
    并行将文件编码为 token IDs，结果存储在 token_id_arr 中
    """
    boundaries = build_line_aligned_boundaries(
        file_path=file_path,
        special_token=b"<|endoftext|>",
        num_chunks= 160
    )

    # 准备每个 chunk 的参数
    # file_path , start_offset, end_offset,token_id_arr,token_id_arr_start_idx= args
    tasks  = []
    for i in range(len(boundaries) - 1):
        tasks.append((
            i,
            file_path,
            boundaries[i],
            boundaries[i + 1],
            token_id_arr, 
            offsets[i],
            counts[i], 
            ))
        
    results = []  #  主进程汇总
    
    # 使用多进程池并行处理
    with Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(vocab_path, merges_path, special_tokens),
    ) as pool:
        for res in tqdm(
                pool.imap_unordered(_encode_tokens_in_chunk_worker, tasks),
                total=len(tasks),
                desc=f"Encoding {file_path.name} to token IDs",
    ):
            results.append(res)
            
    # ========================================================
    # ★ 主进程最终一致性校验
    # ========================================================

    results.sort(key=lambda x: x[0])  # 按 chunk_id 排序

    for chunk_id, written in results:
        if written != counts[chunk_id]:
            raise RuntimeError(
                f"[post-check] chunk {chunk_id}: "
                f"{written} != {counts[chunk_id]}"
            )
       
       
def main():
    num_workers = 6      
    ASSIGNMENT_REPO = Path("/home/fredkeira/projects/assignment1-basics")
    NPY_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/chunk_token_counts.npy"
    OUTPUT_DATA_REPO = ASSIGNMENT_REPO/"token_to_id_outputs"
    
    train_token_arr_path = OUTPUT_DATA_REPO / "train_ids.bin"
    valid_token_arr_path = OUTPUT_DATA_REPO / "valid_ids.bin"
    
    ###### 
    N_train_num = 2_727_120_424
    N_valid_num = 66_401_048

    # 初始化内存映射文件
    train_token_arr = np.memmap(train_token_arr_path,
                                dtype=np.uint16,
                                mode='w+',
                                shape=(N_train_num,)
                                )

    # 160个chunk中的token数量

    counts = np.load(NPY_PATH)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)

    prefix_sum = 0
    offsets[0] = 0
    for i, count in enumerate(counts):
        prefix_sum += count
        offsets[i + 1] = prefix_sum


    # 并行编码训练数据
    parallel_encode_file_to_token_ids(
        file_path=TRAIN_DATA,
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=SPECIAL_TOKENS,
        token_id_arr=train_token_arr,
        num_processes=num_workers,
        offsets=offsets,
        counts=counts,
    )
    
    train_token_arr.flush()

if __name__ == "__main__":
    main()