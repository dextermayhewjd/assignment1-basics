import regex as re

PAT2 = r"""
'(?:[sdmt]|ll|ve|re)
| \ ?\p{L}+
| \ ?\p{N}+
| \ ?[^\s\p{L}\p{N}]+
| \s+(?!\S)
| \s+
"""


text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"


# [('low', 1), (' low', 4), (' lower', 2)
def compute_frequency(text:str)->dict[str,int]:
  
    str_counts: dict[str,int] = {}
    for m in re.finditer(PAT2, text, flags=re.VERBOSE):
        str_counts[m.group(0)] =str_counts.get(m.group(0),0) + 1
    
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
    
    sorted_pair_bytes_dict = sorted(pair_bytes_dict.items(),key=lambda x: (x[1],x[0]),reverse= True) 
    # 这一步取 key (b'la', b'o')
    max_value_key = sorted_pair_bytes_dict[0][0]

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
def train_bpe(text:str,num_merges:int):
    next_id = 256
    vocab : dict[int,bytes] = {x:bytes([x]) for x in range(256)}  
    merges = []
    
    #('low', 1), (' low', 4)
    str_counts: dict[str,int] = compute_frequency(text= text)
    
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
    