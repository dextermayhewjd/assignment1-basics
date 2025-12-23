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

def train_bpe(
              input_path :str,
              vocab_size :int,
              special_tokens:list[str],  
    ):
    """
    Train a byte-level BPE tokenizer.

    Returns
    -------
    vocab: dict[int, bytes]
        token_id -> token_bytes
    merges: list[tuple[bytes, bytes]]
        list of merges in order
    """
    
    def init_vocab(vocab_size:int, special_tokens:list[str]):
        vocab:dict[int,bytes] = { x:bytes[(x)] for x in range(256)}
        next_id = 256
        
        for s in special_tokens:
            if next_id >= vocab_size:
                break
            vocab[next_id] = s.encode("utf-8")
            next_id +=1
        
        return vocab, next_id
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    '''
    读取训练文本
    '''
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()