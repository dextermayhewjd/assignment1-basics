"""
repeat until vocab_size reached:
    1. 统计所有 (token_i, token_{i+1}) pair 的频率
    2. 选出频率最高的 pair
       - 频率相同 → 选 lexicographically 最大的 pair
    3. 把这个 pair merge 成新 token
    4. 更新所有 pre-token 中的序列
"""

def train_bpe(
              input_path :str,
              vocab_size :int,
              sepcial_tokens:list[str],  
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
    
    
    
    
    return 

