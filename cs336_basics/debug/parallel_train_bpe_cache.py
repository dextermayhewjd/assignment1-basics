"""
repeat until vocab_size reached:
    1. 统计所有 (token_i, token_{i+1}) pair 的频率
    2. 选出频率最高的 pair
       - 频率相同 → 选 lexicographically 最大的 pair
    3. 把这个 pair merge 成新 token
    4. 更新所有 pre-token 中的序列
"""

import shutil
import regex as re 
from collections import Counter
from typing import NamedTuple
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from tqdm import trange
from tqdm import tqdm
import time
import tempfile
import pickle
from pathlib import Path

PAT2 = re.compile(r"""
'(?:[sdmt]|ll|ve|re)
| \ ?\p{L}+
| \ ?\p{N}+
| \ ?[^\s\p{L}\p{N}]+
| \s+(?!\S)
| \s+
""", re.VERBOSE)

def init_vocab(vocab_size:int,
               special_tokens:list[str]):
    vocab: dict[int, bytes] = {x: bytes([x]) for x in range(256)}

    next_id = 256
    
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size too small to fit all special tokens")

    for s in special_tokens:
        if next_id >= vocab_size:
            break
        vocab[next_id] = s.encode("utf-8")
        next_id +=1
    
    return vocab, next_id

def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    return re.split(pattern, text)

def compute_frequency_with_special_token(text:str,
                      special_tokens: list[str]
                      )->Counter[str]:
    
    chunks = split_on_special_tokens(text, special_tokens)
        
    str_counts: Counter[str] = Counter()
    for chunk in chunks:
        for m in re.finditer(PAT2, chunk):
            str_counts[m.group(0)] += 1
        
    
    return  str_counts

# (' lower', 2) -> (b'l', b'o', b'w'): 5
def turn_into_bytes_seq_counter(
                           str_counts: Counter[str],
                          )->Counter[tuple[bytes, ...]]:
  
    bytes_seq_counter: Counter[tuple[bytes, ...]] = Counter()
    
    for key,value in str_counts.items():
        bytes_key = key.encode("utf-8")
        tuple_bytes_key = tuple(bytes([x]) for x in bytes_key)
        bytes_seq_counter[tuple_bytes_key] += value

    
    return  bytes_seq_counter
'''
input_path, start, end, special_tokens = args
'''
def parallel_worker(args)->dict[tuple[bytes,...],int]:
    ''' 
    原来是text 现在是 args boundary 要读出来  
    '''
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
    
        #('low', 1), (' low', 4)
        str_counts: Counter[str] = compute_frequency_with_special_token(text= chunk,special_tokens=special_tokens)
        
        #(b'l', b'o', b'w'): 5 因为不跨word 
        bytes_seq_counter: Counter[tuple[bytes,...]] = turn_into_bytes_seq_counter(str_counts=str_counts)

    return bytes_seq_counter
    


#(b'la', b'o', b'w'): 5 ->(b'la', b'o'):5 , (b'o', b'w'):5
#加上 初始化pair2seq 的dict                       
def calculate_pair_bytes_seq_counter(
                            bytes_seq_counter:Counter[tuple[bytes,...]]
                            )->tuple[Counter[tuple[bytes, bytes]],
                                     dict[tuple[bytes,bytes],set[tuple[bytes,...]]]]:
    
    bytes_pair_counter: Counter[tuple[bytes, bytes]] = Counter()
    
    pair2seq_dict:dict[
                    tuple[bytes,bytes],
                    set[tuple[bytes,...]
                                        ]] = defaultdict(set)
    
    for key,value in bytes_seq_counter.items():
        key_len = len(key)
        '''
        当所有 token 序列长度都 ≤ 1 时, pair 统计会返回空字典，训练循环应该停止。
        '''
        for x in range(key_len -1): 
            new_tuple = (key[x],key[x+1])
            bytes_pair_counter[new_tuple] +=value
            pair2seq_dict[new_tuple].add(key)
    return bytes_pair_counter, pair2seq_dict

# (b'l', b'o'):5,(b'a', b'b'):9,(b'c', b'd'):5  ->  (b'a', b'b')
def get_highest_pair(
                bytes_pair_counter:Counter[tuple[bytes, bytes]]
                )->tuple[bytes,bytes]:
    
    # sorted_pair_bytes_dict = sorted(pair_bytes_dict.items(),key=lambda x: (x[1],x[0]),reverse= True) 
    # # 这一步取 key (b'la', b'o')
    # max_value_key = sorted_pair_bytes_dict[0][0]
    
    # 这个版本更快
    max_value_key = max(bytes_pair_counter.items(), key=lambda kv: (kv[1], kv[0]))[0]


    return max_value_key

def merge_pair_in_token(token:tuple[bytes,...],
                        pair:tuple[bytes,bytes]
                        )-> tuple[bytes,...]:
    merged = pair[0]+pair[1]
    result = []
    i = 0
    while i < len(token):
        if i<len(token) - 1 and token[i] == pair[0] and token[i+1] == pair[1]:
            result.append(merged)
            i += 2
        else:
            result.append(token[i])
            i +=1
    return tuple(result)
'''
重点想好哪里开始更新
应该是每次更新的是bytes_seq_counter
'''

class BPETrainResult(NamedTuple):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    """
    vocab: dict[int, bytes]
        token_id -> token_bytes
    merges: list[tuple[bytes, bytes]]
        list of merges in order
    """
    
    
def train_bpe(
              input_path :str,
              vocab_size :int,
              special_tokens:list[str],  
    ):

    # =========================
    # 【新增】计时
    # =========================
    print("start train_bpe")
    start_time = time.time()
    
    vocab, next_id = init_vocab(
        vocab_size= vocab_size,
        special_tokens=special_tokens
    )
    
    merges = []
    initial_vocab_size = len(vocab)
    num_merges = vocab_size - initial_vocab_size


    '''
    首先这里读的是bytes了
    '''
    with open(input_path, "rb") as f:
        # 应该是20
        num_processes = 10
        
        # [100,200,300,400,500,600]
        print("start to find boundaries")
        boundaries = find_chunk_boundaries(f, num_processes*20, b"<|endoftext|>")
        print("boundaries founded")
        
        # print("\n=== CHUNK BOUNDARIES ===")
        # print(f"num boundaries: {len(boundaries)}  (→ {len(boundaries)-1} chunks)")
        # for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        #     print(f"chunk {i:02d}: [{s:,} → {e:,}]  size={(e-s)/1024/1024:.2f} MB")
        # print("=========================\n")
        
        '''
        zip(
                [0, 100, 230],
                [100, 230, 400]
            )
        (0,100), (100,230), (230,400)
        '''
        worker_parameters:list = [(input_path,start,end,special_tokens)for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        # 这个是原来的dict版本现在换成Counter 可以直接加
        # bytes_seq_counter: dict[tuple[bytes,...],int] = {}

#12-25z 之前的1        
        # bytes_seq_counter:Counter[tuple[bytes,...]] = Counter()
        # with Pool(num_processes) as pool:
        #       sub_counters = pool.map(parallel_worker,worker_parameters)
        # bytes_seq_counter = sum(sub_counters,Counter())
        
#12-25z 之后的2               
        # bytes_seq_counter:Counter[tuple[bytes,...]] = Counter()
        # with Pool(num_processes) as pool:
        #     for sub in tqdm(
        #         pool.imap_unordered(parallel_worker, worker_parameters, chunksize=1),
        #         total=len(worker_parameters),
        #         desc="Pretokenize chunks",
        #     ):
        #         bytes_seq_counter += sub
# 使用mapreduce方式 处理 3
        
        tmp_dir = Path(tempfile.mkdtemp(prefix="bpe_mapreduce_"))
        print(f"[MapReduce] spill directory: {tmp_dir}")

        paths = []        # 记录所有 map 输出文件路径

        # ---------------- MAP（并行）----------------
        with Pool(num_processes) as pool:
            for i, sub in enumerate(
                tqdm(
                    pool.imap_unordered(parallel_worker, worker_parameters, chunksize=1),
                    total=len(worker_parameters),
                    desc="Pretokenize (map)"
                )
            ):
                path = tmp_dir / f"part_{i:04d}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(sub, f, protocol=pickle.HIGHEST_PROTOCOL)
                paths.append(path)

        # ---------------- REDUCE（串行、流式）----------------
        bytes_seq_counter: Counter[tuple[bytes, ...]] = Counter()

        BATCH = 16        # 每 16 份做一次“局部 reduce”
        batch = []

        for path in tqdm(paths, desc="Reduce"):
            with open(path, "rb") as f:
                batch.append(pickle.load(f))

            if len(batch) >= BATCH:
                # 局部 reduce（树状）
                local = sum(batch, Counter())
                bytes_seq_counter += local
                batch.clear()

        # 剩余的再收一次
        if batch:
            local = sum(batch, Counter())
            bytes_seq_counter += local

        # （可选）清理 tmp
        shutil.rmtree(tmp_dir)        
                
        # 初始化  (b'l', b'o'):5
        bytes_pair_counter: Counter[tuple[bytes, bytes]]
        bytes_pair_counter,pair2seq_dict = calculate_pair_bytes_seq_counter(bytes_seq_counter = bytes_seq_counter)
        
        print("start to merge")
        # for _ in range(num_merges):
            #(b'l', b'o'):5
        for merge_idx in trange(
            num_merges,
            desc="BPE merges",
            unit="merge",
        ):    
            # pair2seq_dict: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]

            if not bytes_pair_counter:
                break  # 没有任何pair可merge了  
            highest_pair:tuple[bytes,bytes] = get_highest_pair(bytes_pair_counter=bytes_pair_counter)
            # set [bytes_seq_counter,...]
            afftected_token = pair2seq_dict.get(highest_pair, set())
            
            a, b = highest_pair # b'a' , b'b'
            merged_bytes = a + b  # b'ab'
            
            
            vocab[next_id] = merged_bytes
            merges.append(highest_pair)
            next_id += 1
            
            '''
            首先 merge 只merge 需要merge的
            merge 会怎么影响 bytes_seq_counter  (b'la', b'o', b'w'): 5 ->(b'la', b'o'):5
                            和bytes_pair_counter(b'l', b'o'):5
            '''
            for token in list(afftected_token):
                    freq = bytes_seq_counter[token]
                    i = 0
                    new_token = []                   
                    token_length = len(token)
                    
                    while i < token_length:
                        if i < token_length - 1 and token[i] == a and token[i+1] == b:
                            new_token.append(merged_bytes)
                            i += 2
                        else:
                            new_token.append(token[i])
                            i += 1
                            
                    old_token = token
                    new_token = tuple(new_token)    
                    
                    for j in range(len(old_token) - 1):
                        p = (old_token[j], old_token[j+1])
                        bytes_pair_counter[p] -= freq
                        if bytes_pair_counter[p] == 0:
                            bytes_pair_counter.pop(p)

                    for j in range(len(new_token) - 1):
                        p = (new_token[j], new_token[j+1])
                        bytes_pair_counter[p] += freq        
                    
                    #  bytes_seq_counter
                    bytes_seq_counter.pop(old_token)
                    bytes_seq_counter[new_token] += freq
                    
                    #  从 pair2seq_dict 移除 old token
                    for j in range(len(old_token) - 1):
                        pair = (old_token[j], old_token[j+1])
                        seq_set = pair2seq_dict.get(pair)
                        if seq_set is not None:
                            seq_set.discard(old_token)

                        # 如果全局计数里已经没有它了（被 pop 或为 0），并且集合也空，就删 key
                        if (pair not in bytes_pair_counter or bytes_pair_counter.get(pair, 0) == 0) and not seq_set:
                            pair2seq_dict.pop(pair, None)    
                    
                    #  加入 new token 的所有 pair
                    for j in range(len(new_token) - 1):
                        pair = (new_token[j], new_token[j + 1])
                        pair2seq_dict.setdefault(pair, set()).add(new_token)
                        
            if highest_pair not in bytes_pair_counter or bytes_pair_counter.get(highest_pair, 0) == 0:
                pair2seq_dict.pop(highest_pair, None)
                        
            # ==================================================
            # 【新增】更有信息量的统计（每 500 次）
            # ==================================================
            if merge_idx % 500 == 0 and merge_idx > 0:
                longest = max(len(tok) for tok in vocab.values())
                top_freq = bytes_pair_counter.get(highest_pair, 0)

                print(
                    f"\n[BPE] merge {merge_idx}/{num_merges} | "
                    f"vocab={len(vocab)} | "
                    f"longest_token_bytes={longest} | "
                    f"top_pair_freq={top_freq}"
                )
                
        # =========================
        # 【新增】训练结束统计
        # =========================
        elapsed = time.time() - start_time
        print(f"  total time: {elapsed / 60:.2f} minutes")
        
        return  vocab, merges
