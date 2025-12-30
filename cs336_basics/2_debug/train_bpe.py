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
                      )->dict[str,int]:
    
    PAT2 = r"""
    '(?:[sdmt]|ll|ve|re)
    | \ ?\p{L}+
    | \ ?\p{N}+
    | \ ?[^\s\p{L}\p{N}]+
    | \s+(?!\S)
    | \s+
    """
    chunks = split_on_special_tokens(text, special_tokens)
    
    str_counts: dict[str,int] = {}
    
    for chunk in chunks:
        for m in re.finditer(PAT2, chunk, flags=re.VERBOSE):
            tok = m.group(0)
            str_counts[tok] = str_counts.get(tok, 0) + 1
    
    return  str_counts
# (' lower', 2) -> (b'l', b'o', b'w'): 5
def turn_into_bytes_count(str_counts:dict[str,int])->dict[tuple[bytes,...],int]:
    bytes_counts: dict[tuple[bytes,...],int] = {}
    
    for key,value in str_counts.items():
        bytes_key = key.encode("utf-8")
        tuple_bytes_key = tuple(bytes([x]) for x in bytes_key)
        bytes_counts[tuple_bytes_key] = bytes_counts.get(tuple_bytes_key, 0) + value

    
    return  bytes_counts

#(b'la', b'o', b'w'): 5 ->(b'la', b'o'):5 , (b'o', b'w'):5
def calculate_pair_bytes_count(bytes_counts:dict[tuple[bytes,...],int])->dict[tuple[bytes,bytes],int]:
    pair_bytes_dict = {}
    for key,value in bytes_counts.items():
        key_len = len(key)
        '''
        当所有 token 序列长度都 ≤ 1 时, pair 统计会返回空字典，训练循环应该停止。
        '''
        for x in range(key_len -1): 
            new_tuple = (key[x],key[x+1])
            pair_bytes_dict[new_tuple] = pair_bytes_dict.get(new_tuple,0) + value
    
    return pair_bytes_dict

# (b'l', b'o'):5,(b'a', b'b'):9,(b'c', b'd'):5  ->  (b'a', b'b')
def get_highest_pair(pair_bytes_dict:dict[tuple[bytes,bytes],int])->tuple[bytes,bytes]:
    
    # sorted_pair_bytes_dict = sorted(pair_bytes_dict.items(),key=lambda x: (x[1],x[0]),reverse= True) 
    # # 这一步取 key (b'la', b'o')
    # max_value_key = sorted_pair_bytes_dict[0][0]
    
    # 这个版本更快
    max_value_key = max(pair_bytes_dict.items(), key=lambda kv: (kv[1], kv[0]))[0]


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
    # 假设 next_id 是256 说明已经有0-255 256 个了
    # 已知 vocab是256的话 停下来的就是next_id = vocab
    # 假设 vocab是257的话 那么 257-256 = merge次数 
    # vocab - next_id 
    vocab, next_id = init_vocab(vocab_size= vocab_size,special_tokens=special_tokens)
    merges = []
    initial_vocab_size = len(vocab)
    num_merges = vocab_size - initial_vocab_size

    '''
    读取训练文本
    '''
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
        
        #('low', 1), (' low', 4)
        str_counts: dict[str,int] = compute_frequency_with_special_token(text= text,special_tokens=special_tokens)
        
        #(b'l', b'o', b'w'): 5 因为不跨word 
        bytes_counts: dict[tuple[bytes,...],int] = turn_into_bytes_count(str_counts=str_counts)
        
        for _ in range(num_merges):
            #(b'l', b'o'):5
            pair_bytes_dict:dict[tuple[bytes,bytes],int] = calculate_pair_bytes_count(bytes_counts = bytes_counts)
            if not pair_bytes_dict:
                break  # 没有任何pair可merge了  
            highest_pair:tuple[bytes,bytes] = get_highest_pair(pair_bytes_dict=pair_bytes_dict)
            
            merged_bytes = highest_pair[0] + highest_pair[1]
            vocab[next_id] = merged_bytes
            merges.append(highest_pair)
            next_id += 1
                
            new_tuple_bytes_counts = {}
            for tuple_bytes,value in bytes_counts.items():
                new_tuple_bytes = merge_pair_in_token(token=tuple_bytes,pair= highest_pair)
                new_tuple_bytes_counts[new_tuple_bytes] = new_tuple_bytes_counts.get(new_tuple_bytes,0) + value

            bytes_counts = new_tuple_bytes_counts
        return  vocab, merges
