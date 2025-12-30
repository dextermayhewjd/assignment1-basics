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

str_counts: dict[str,int] = {}

# task1 统计词频 
for m in re.finditer(PAT2, text, flags=re.VERBOSE):
    str_counts[m.group(0)] =str_counts.get(m.group(0),0) + 1

# task 2 创建 vocab 的字典
vocab : dict[int,bytes] = {x:bytes([x]) for x in range(256)}
#  新的token id 是256开始 因为0到255 都有 长度为1的单byte bytes了
max_index = 256


# task3 
# dict_items([('low', 1), (' low', 4), (' lower', 2), (' widest', 3), (' newest', 6)])
# 目标 (b'l', b'o', b'w'): 5
bytes_counts: dict[tuple[bytes,...],int] = {}

for key,value in str_counts.items():
    bytes_key = key.encode("utf-8")
    tuple_bytes_key = tuple(bytes([x]) for x in bytes_key)
    bytes_counts[tuple_bytes_key] = value
    
# task4  
# eg. (b'l', b'o', b'w'): 5 ->(b'l', b'o'):5
# 生成pair 这一步日后会出问题吗？
# 不会 因为 (b'la', b'o', b'w'): 5 ->(b'la', b'o'):5
pair_bytes_dict = {}
for key,value in bytes_counts.items():
    key_len = len(key)
    for x in range(key_len -1):
        new_tuple = (key[x],key[x+1])
        pair_bytes_dict[new_tuple] = pair_bytes_dict.get(new_tuple,0) + value
        
sorted_pair_bytes_dict = sorted(pair_bytes_dict.items(),key=lambda x: (x[1],x[0]),reverse= True) 
# 这一步取 key (b'la', b'o')
max_value_key = sorted_pair_bytes_dict[0][0]

# task 这个是单一token merge 
# 要在 所有的token里找这一个token
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
  
# 在哪里找一群要merge的token呢
# out_put 这里 对应的是 统计完词的频次后 刚生成的dict[tuple[bytes,..] ： int] 
new_tuple_bytes_counts = {}
for tuple_bytes,value in bytes_counts.items():
    new_tuple_bytes = merge_pair_in_token(token=tuple_bytes,pair= max_value_key)
    new_tuple_bytes_counts[new_tuple_bytes] = new_tuple_bytes_counts.get(new_tuple_bytes,0) + value

bytes_counts = new_tuple_bytes_counts