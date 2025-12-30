
from collections import defaultdict
from collections import Counter
test_dict_str_num = {'low': 5, 'lower': 2, 'widest': 3, 'newest': 6}


def get_highest_pair(
                pair_bytes_Counter:Counter[tuple[bytes, bytes]]
                )->tuple[bytes,bytes]:
    
    max_value_key = max(pair_bytes_Counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return max_value_key
'''
初始化bytes_seq_counter 
'''
bytes_seq_counter = Counter()
for key,value in test_dict_str_num.items():
    bytes_key = key.encode("utf-8")
    tuple_bytes_key = tuple(bytes([x]) for x in bytes_key)
    bytes_seq_counter[tuple_bytes_key] = value

# print(bytes_seq_counter)

 
# bytes_seq_counter =  Counter({
# (b'n', b'e', b'w', b'e', b's', b't'): 6,
# (b'l', b'o', b'w'): 5,
# (b'w', b'i', b'd', b'e', b's', b't'): 3,
# (b'l', b'o', b'w', b'e', b'r'): 2
# })
'''
如何在初始化 bytes_pair_counter
的同时 初始化 pair2seq_dict dict[pair,seq]
'''

bytes_pair_counter: Counter[tuple[bytes,bytes], int] 
bytes_pair_counter = Counter()

pair2seq_dict:dict[
                  tuple[bytes,bytes],
                  set[tuple[bytes,...]
                                    ]] = defaultdict(set)
'''
注意这里是set 所以 byte_seq
'''
for key,value in bytes_seq_counter.items():
#     # key  (b'l', b'o', b'w')
#     # print(f'key = {key}')
#     # print(f'length of key ={len(key)}')
#     # print(value)
    key_len = len(key)
    for x in range(key_len -1):
        # print(f'{key[x]}+ {key[x+1]}')
        '''
        应为此处是tuple 所以 key[x] 是bytes 而不是 整数
        '''
        new_tuple = (key[x],key[x+1])
        bytes_pair_counter[new_tuple] += value
        pair2seq_dict[new_tuple].add(key)
'''
bytes_pair_counter

Counter({
(b'e', b's'): 9, 
(b's', b't'): 9, 
(b'w', b'e'): 8, 
(b'l', b'o'): 7, 
(b'o', b'w'): 7, 
(b'n', b'e'): 6, 
(b'e', b'w'): 6, 
(b'w', b'i'): 3, 
(b'i', b'd'): 3, 
(b'd', b'e'): 3, 
(b'e', b'r'): 2}
)

pair2seq_dict

((b'l', b'o'), {(b'l', b'o', b'w'), (b'l', b'o', b'w', b'e', b'r')})
((b'o', b'w'), {(b'l', b'o', b'w'), (b'l', b'o', b'w', b'e', b'r')})
((b'w', b'e'), {(b'n', b'e', b'w', b'e', b's', b't'), (b'l', b'o', b'w', b'e', b'r')})
((b'e', b'r'), {(b'l', b'o', b'w', b'e', b'r')})
((b'w', b'i'), {(b'w', b'i', b'd', b'e', b's', b't')})
((b'i', b'd'), {(b'w', b'i', b'd', b'e', b's', b't')})
((b'd', b'e'), {(b'w', b'i', b'd', b'e', b's', b't')})
((b'e', b's'), {(b'w', b'i', b'd', b'e', b's', b't'), (b'n', b'e', b'w', b'e', b's', b't')})
((b's', b't'), {(b'w', b'i', b'd', b'e', b's', b't'), (b'n', b'e', b'w', b'e', b's', b't')})
((b'n', b'e'), {(b'n', b'e', b'w', b'e', b's', b't')})
((b'e', b'w'), {(b'n', b'e', b'w', b'e', b's', b't')})
'''
highest_pair:tuple[bytes,bytes] = get_highest_pair(pair_bytes_Counter=bytes_pair_counter)

a, b = highest_pair # b'a' , b'b'
merged_bytes = a + b  # b'ab'


'''
highest_pair
(b's', b't')
'''

# set [bytes_seq_counter,...]
afftected_token = pair2seq_dict.get(highest_pair, set())
'''
{(b'w', b'i', b'd', b'e', b's', b't'), # windest 
(b'n', b'e', b'w', b'e', b's', b't')} # newest
'''
for token in list(afftected_token):
    freq = bytes_seq_counter[token]
    
    i = 0 # original_token_index

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
        pair2seq_dict[(old_token[j], old_token[j+1])].discard(old_token)    
    
    #  加入 new token 的所有 pair
    for j in range(len(new_token) - 1):
        pair2seq_dict[(new_token[j], new_token[j+1])].add(new_token)
    
    