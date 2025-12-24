"""
repeat until vocab_size reached:
    1. 统计所有 (token_i, token_{i+1}) pair 的频率
    2. 选出频率最高的 pair
       - 频率相同 → 选 lexicographically 最大的 pair
    3. 把这个 pair merge 成新 token
    4. 更新所有 pre-token 中的序列
"""

import regex as re 
from collections import Counter
from typing import NamedTuple
from cs336_basics.pretokenization_example import find_chunk_boundaries
from multiprocessing import Pool, cpu_count
from collections import defaultdict
from tqdm import trange
import time

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
    
    # str_counts: dict[str,int] = {}
    
    # for chunk in chunks:
    #     for m in re.finditer(PAT2, chunk, flags=re.VERBOSE):
    #         tok = m.group(0)
    #         str_counts[tok] = str_counts.get(tok, 0) + 1
    
    str_counts: Counter[str] = Counter()
    for chunk in chunks:
        for m in re.finditer(PAT2, chunk):
            str_counts[m.group(0)] += 1
        
    
    return  str_counts
# (' lower', 2) -> (b'l', b'o', b'w'): 5
def turn_into_bytes_count(
                           str_counts: Counter[str],
                          )->Counter[tuple[bytes, ...]]:
  
    bytes_counts: Counter[tuple[bytes, ...]] = Counter()
    
    for key,value in str_counts.items():
        bytes_key = key.encode("utf-8")
        tuple_bytes_key = tuple(bytes([x]) for x in bytes_key)
        bytes_counts[tuple_bytes_key] += value

    
    return  bytes_counts
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
        bytes_counts: Counter[tuple[bytes,...]] = turn_into_bytes_count(str_counts=str_counts)

    return bytes_counts
    


#(b'la', b'o', b'w'): 5 ->(b'la', b'o'):5 , (b'o', b'w'):5
#加上 初始化pair2seq 的dict                       
def calculate_pair_bytes_count(
                            bytes_counts:Counter[tuple[bytes,...]]
                            )->Counter[tuple[bytes, bytes]]:
    
    pair_counts: Counter[tuple[bytes, bytes]] = Counter()

    for key,value in bytes_counts.items():
        key_len = len(key)
        '''
        当所有 token 序列长度都 ≤ 1 时, pair 统计会返回空字典，训练循环应该停止。
        '''
        for x in range(key_len -1): 
            new_tuple = (key[x],key[x+1])
            pair_counts[new_tuple] +=value
            # pair2seq_dict[new_tuple].add(key)
    return pair_counts

# (b'l', b'o'):5,(b'a', b'b'):9,(b'c', b'd'):5  ->  (b'a', b'b')
def get_highest_pair(
                pair_bytes_Counter:Counter[tuple[bytes, bytes]]
                )->tuple[bytes,bytes]:
    
    # sorted_pair_bytes_dict = sorted(pair_bytes_dict.items(),key=lambda x: (x[1],x[0]),reverse= True) 
    # # 这一步取 key (b'la', b'o')
    # max_value_key = sorted_pair_bytes_dict[0][0]
    
    # 这个版本更快
    max_value_key = max(pair_bytes_Counter.items(), key=lambda kv: (kv[1], kv[0]))[0]


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
应该是每次更新的是bytes_counts
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
        num_processes = cpu_count()
        
        # [100,200,300,400,500,600]
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        '''
        zip(
                [0, 100, 230],
                [100, 230, 400]
            )
        (0,100), (100,230), (230,400)
        '''
        worker_parameters:list = [(input_path,start,end,special_tokens)for start, end in zip(boundaries[:-1], boundaries[1:])]
        
        # 这个是原来的dict版本现在换成Counter 可以直接加
        # bytes_counts: dict[tuple[bytes,...],int] = {}
        
        bytes_counts:Counter[tuple[bytes,...]] = Counter()
        with Pool(num_processes) as pool:
              sub_counters = pool.map(parallel_worker,worker_parameters)
        bytes_counts = sum(sub_counters,Counter())
        
        
        # 初始化  (b'l', b'o'):5
        pair_bytes_Counter: Counter[tuple[bytes, bytes]]
        pair_bytes_Counter = calculate_pair_bytes_count(bytes_counts = bytes_counts)
        
        
        # for _ in range(num_merges):
            #(b'l', b'o'):5
        for merge_idx in trange(
            num_merges,
            desc="BPE merges",
            unit="merge",
        ):    
            # pair2seq_dict: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]

            if not pair_bytes_Counter:
                break  # 没有任何pair可merge了  
            highest_pair:tuple[bytes,bytes] = get_highest_pair(pair_bytes_Counter=pair_bytes_Counter)
            
            a, b = highest_pair
            merged_bytes = a + b 
            
            
            vocab[next_id] = merged_bytes
            merges.append(highest_pair)
            next_id += 1
            
            '''
            首先 merge 只merge 需要merge的
            merge 会怎么影响 bytes_counts  (b'la', b'o', b'w'): 5 ->(b'la', b'o'):5
                            和pair_bytes_Counter(b'l', b'o'):5
            '''
            for token, freq in list(bytes_counts.items()):
                    i = 0
                    new_token = []
                    changed = False

                    while i < len(token):
                        if i < len(token) - 1 and token[i] == a and token[i+1] == b:
                            left = token[i-1] if i > 0 else None
                            right = token[i+2] if i + 2 < len(token) else None

                            # 1️⃣ 减旧 pair
                            pair_bytes_Counter[(a, b)] -= freq
                            if left is not None:
                                pair_bytes_Counter[(left, a)] -= freq
                                pair_bytes_Counter[(left, merged_bytes)] += freq
                            if right is not None:
                                pair_bytes_Counter[(b, right)] -= freq
                                pair_bytes_Counter[(merged_bytes, right)] += freq

                            # 2️⃣ 再真的 merge
                            new_token.append(merged_bytes)
                            i += 2
                            changed = True
                            
                        else:
                            new_token.append(token[i])
                            i +=1
                    if changed:
                        bytes_counts.pop(token)
                        bytes_counts[tuple(new_token)] += freq
            # ==================================================
            # 【新增】更有信息量的统计（每 500 次）
            # ==================================================
            if merge_idx % 500 == 0 and merge_idx > 0:
                longest = max(len(tok) for tok in vocab.values())
                top_freq = pair_bytes_Counter.get(highest_pair, 0)

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
        print("\n[BPE DONE]")
        print(f"  merges learned: {len(merges)}")
        print(f"  vocab size: {len(vocab)}")
        print(f"  total time: {elapsed / 60:.2f} minutes")
        
        return  vocab, merges
