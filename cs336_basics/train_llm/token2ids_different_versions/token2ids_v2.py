'''
Docstring for cs336_basics.train_llm.token2ids_different_versions.token2ids
最开始的实验方式
尝试多worker 多写入 
但是for loop 调用np.memmap写入时 token级别写入

'''


from pathlib import Path
import multiprocessing as mp
from tqdm import tqdm

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
# Globals (per worker process)
# ============================================================

_GLOBAL_TOKENIZER: Tokenizer | None = None
_GLOBAL_QUEUE = None  # mp.Queue


def _init_worker(vocab_path: Path, merges_path: Path, special_tokens: list[str], out_queue):
    """
    每个 worker 进程启动时调用一次：
    - 初始化 tokenizer
    - 保存全局 queue 句柄
    """
    global _GLOBAL_TOKENIZER, _GLOBAL_QUEUE
    _GLOBAL_TOKENIZER = Tokenizer.from_files(
        vocab_filepath=vocab_path,
        merges_filepath=merges_path,
        special_tokens=special_tokens,
    )
    _GLOBAL_QUEUE = out_queue


# ============================================================
# Chunk boundary helpers (保持你原逻辑)
# ============================================================

def align_boundary_to_line_start(f, offset: int) -> int:
    """将 byte offset 对齐到所在行的行首"""
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
    """
    1) special token 粗切
    2) 对齐到行首
    3) 去重排序
    """
    with open(file_path, "rb") as f:
        rough_boundaries = find_chunk_boundaries(
            f,
            desired_num_chunks=num_chunks,
            split_special_token=special_token,
        )
        line_aligned = [align_boundary_to_line_start(f, off) for off in rough_boundaries]

    aligned = sorted(set(line_aligned))
    return aligned


# ============================================================
# Writer process
# ============================================================

def _writer_process(
    queue: mp.Queue,
    out_bin_path: str,
    dtype_str: str,
    total_tokens: int,
    counts: np.ndarray,
    num_chunks: int,
):
    """
    单 writer：唯一写入 memmap 的进程
    消息协议：
      ("DATA", chunk_id, start_idx, np_array_uint16)
      ("END",  chunk_id, written_count)
    """
    dtype = np.dtype(dtype_str)
    mm = np.memmap(out_bin_path, dtype=dtype, mode="r+", shape=(total_tokens,))

    finished = 0
    written_by_chunk = np.zeros(num_chunks, dtype=np.int64)

    while True:
        msg = queue.get()
        tag = msg[0]

        if tag == "DATA":
            _, chunk_id, start_idx, arr = msg
            # arr 是 np.ndarray(dtype=uint16) 的一个块
            end_idx = start_idx + arr.shape[0]
            mm[start_idx:end_idx] = arr  # 大块写
        elif tag == "END":
            _, chunk_id, written = msg
            written_by_chunk[chunk_id] = written
            finished += 1

            if finished == num_chunks:
                break
        else:
            raise RuntimeError(f"Unknown message tag: {tag}")

    # flush + 校验
    mm.flush()

    # 最终一致性校验：每个 chunk 写入 token 数必须等于 counts
    for i in range(num_chunks):
        if written_by_chunk[i] != counts[i]:
            raise RuntimeError(
                f"[writer-check] chunk {i}: expected {counts[i]}, got {written_by_chunk[i]}"
            )


# ============================================================
# Worker: tokenize chunk -> send blocks to writer
# ============================================================

def _encode_tokens_in_chunk_worker(args):
    """
    每个 worker 对指定 chunk tokenize，然后通过 queue 发送给 writer
    """
    (
        chunk_id,
        file_path,
        start_offset,
        end_offset,
        chunk_token_start_idx,  # offsets[chunk_id]
        expected_count,
        dtype_str,
        flush_tokens,           # 每多少 token 发一次
    ) = args

    tokenizer = _GLOBAL_TOKENIZER
    queue = _GLOBAL_QUEUE
    dtype = np.dtype(dtype_str)

    # 写入位置（token 索引）
    write_pos = int(chunk_token_start_idx)
    emitted = 0

    buf = []  # list[int]

    with open(file_path, "rb") as f:
        f.seek(start_offset)

        # 行级读取保持不变（你不想改 chunk 逻辑）
        while f.tell() < end_offset:
            line = f.readline()
            if not line:
                break

            text = line.decode("utf-8", errors="ignore")

            # 这里避免 encode_iterable([text]) 的额外包装也行：
            # 但不知道你 Tokenizer 是否有 encode(text)->list[int]，所以保守用 encode_iterable
            for tid in tokenizer.encode_iterable([text]):
                buf.append(tid)
                emitted += 1

                # 达到阈值就 flush 一次（防止 worker 内存累积）
                if len(buf) >= flush_tokens:
                    arr = np.asarray(buf, dtype=dtype)
                    queue.put(("DATA", chunk_id, write_pos, arr))
                    write_pos += arr.shape[0]
                    buf.clear()

    # flush tail
    if buf:
        arr = np.asarray(buf, dtype=dtype)
        queue.put(("DATA", chunk_id, write_pos, arr))
        write_pos += arr.shape[0]
        buf.clear()

    # chunk-level 校验（worker 自己先做一次）
    if emitted != expected_count:
        raise RuntimeError(
            f"[chunk {chunk_id}] token mismatch: expected {expected_count}, got {emitted}"
        )

    # 通知 writer 本 chunk 完成
    queue.put(("END", chunk_id, emitted))

    return chunk_id, emitted


# ============================================================
# Orchestrator
# ============================================================

def parallel_encode_file_to_token_ids_single_writer(
    file_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    out_bin_path: Path,          # writer 打开 memmap 的文件路径
    dtype: np.dtype,
    total_tokens: int,           # memmap shape
    num_processes: int,
    offsets: np.ndarray,         # chunk -> token start idx
    counts: np.ndarray,          # chunk -> token count
    flush_tokens: int = 200_000, # 每个 block 的 token 数上限（控制内存/吞吐）
    queue_maxsize: int | None = None,
):
    """
    160 chunks 固定：
    - boundaries 仍然用 special token + line align
    - tasks 仍然一 chunk 一个
    - worker tokenize -> queue -> writer 写 memmap
    """
    boundaries = build_line_aligned_boundaries(
        file_path=file_path,
        special_token=b"<|endoftext|>",
        num_chunks=160,
    )

    num_chunks = len(boundaries) - 1
    if num_chunks != 160:
        # 你说你提前切了 160 个；这里防御一下
        raise RuntimeError(f"Expected 160 chunks, got {num_chunks}")

    # queue 容量：不给也行，但建议 bounded 防止 writer 跟不上导致内存涨
    if queue_maxsize is None:
        queue_maxsize = max(4 * num_processes, 64)

    ctx = mp.get_context("fork")  # Linux 默认 fork；如需更安全可改 spawn（但会慢）
    q: mp.Queue = ctx.Queue(maxsize=queue_maxsize)

    # 确保输出文件存在并且大小正确（提前创建 memmap 文件）
    # 注意：writer 用 r+ 模式要求文件已存在且大小足够
    mm = np.memmap(out_bin_path, dtype=dtype, mode="w+", shape=(total_tokens,))
    mm.flush()
    del mm
    
    dtype = np.dtype(dtype)   
    # 启动 writer
    writer = ctx.Process(
        target=_writer_process,
        args=(q, str(out_bin_path), dtype.str, total_tokens, counts, num_chunks),
        daemon=False,
    )
    writer.start()

    # tasks：chunk 固定一个任务
    tasks = []
    for i in range(num_chunks):
        tasks.append((
            i,
            file_path,
            boundaries[i],
            boundaries[i + 1],
            int(offsets[i]),
            int(counts[i]),
            dtype.str,
            int(flush_tokens),
        ))

    # pool workers
    results = []
    with ctx.Pool(
        processes=num_processes,
        initializer=_init_worker,
        initargs=(vocab_path, merges_path, special_tokens, q),
    ) as pool:
        for res in tqdm(
            pool.imap_unordered(_encode_tokens_in_chunk_worker, tasks),
            total=len(tasks),
            desc=f"Tokenizing {file_path.name}",
        ):
            results.append(res)

    # 等 writer 结束（writer 自己校验 counts）
    writer.join()
    if writer.exitcode != 0:
        raise RuntimeError(f"writer process failed with exitcode={writer.exitcode}")

    # 主进程再做一次轻量校验（可选）
    results.sort(key=lambda x: x[0])
    for chunk_id, written in results:
        if written != counts[chunk_id]:
            raise RuntimeError(
                f"[post-check] chunk {chunk_id}: expected {counts[chunk_id]}, got {written}"
            )


# ============================================================
# main
# ============================================================

def main():
    num_workers = 20      
    ASSIGNMENT_REPO = Path("/home/fredkeira/projects/assignment1-basics")
    
    OWT_TRAIN_NPY_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/scripts/owt_train_chunk_token_counts.npy"
    OWT_VALID_NPY_PATH = ASSIGNMENT_REPO / "cs336_basics/train_llm/scripts/owt_valid_chunk_token_counts.npy"
    
    OUTPUT_DATA_REPO = ASSIGNMENT_REPO/"token_to_id_outputs"
    
    # train_token_arr_path = OUTPUT_DATA_REPO / "train_ids.bin"
    valid_token_arr_path = OUTPUT_DATA_REPO / "valid_ids.bin"
    
    ###### 
    N_train_num = 2_727_120_424
    N_valid_num = 66_401_048


    # counts = np.load(OWT_TRAIN_NPY_PATH)
    # offsets = np.zeros(len(counts) + 1, dtype=np.int64)

    # prefix_sum = 0
    # offsets[0] = 0
    # for i, count in enumerate(counts):
    #     prefix_sum += count
    #     offsets[i + 1] = prefix_sum
        
    # # offsets[i] 是 chunk i 的 token 起点，counts[i] 是 chunk i 的 token 数
    # # total_tokens 必须等于 offsets[-1]
    # if int(offsets[-1]) != N_train_num:
    #     raise RuntimeError(f"offsets[-1]={offsets[-1]} != N_train_num={N_train_num}")


    
    

    counts = np.load(OWT_VALID_NPY_PATH)
    offsets = np.zeros(len(counts) + 1, dtype=np.int64)

    prefix_sum = 0
    offsets[0] = 0
    for i, count in enumerate(counts):
        prefix_sum += count
        offsets[i + 1] = prefix_sum
        
    # offsets[i] 是 chunk i 的 token 起点，counts[i] 是 chunk i 的 token 数
    # total_tokens 必须等于 offsets[-1]
    if int(offsets[-1]) != N_valid_num:
        raise RuntimeError(f"offsets[-1]={offsets[-1]} != N_valid_num={N_valid_num}")


    # # 并行编码训练数据
    # parallel_encode_file_to_token_ids(
    #     file_path=TRAIN_DATA,
    #     vocab_path=VOCAB_PATH,
    #     merges_path=MERGES_PATH,
    #     special_tokens=SPECIAL_TOKENS,
    #     token_id_arr=train_token_arr,
    #     num_processes=num_workers,
    #     offsets=offsets,
    #     counts=counts,
    # )
    
    # train_token_arr.flush()

    # 并行编码验证数据
    parallel_encode_file_to_token_ids_single_writer(
        file_path=VALID_DATA,
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=SPECIAL_TOKENS,
        out_bin_path=valid_token_arr_path,
        dtype=np.uint16,
        total_tokens=N_valid_num,
        num_processes=num_workers,
        offsets=offsets,
        counts=counts,
        flush_tokens=100_000,      # 你可以调大到 200k/500k；越大越省 queue/IPC，但单块更大
        queue_maxsize=128,         # bounded 防止内存涨
    )



if __name__ == "__main__":
    main()