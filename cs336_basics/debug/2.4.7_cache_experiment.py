
from collections import defaultdict
test_dict_str_num = {'low': 5, 'lower': 2, 'widest': 3, 'newest': 6}

output_dict = {}
for key,value in test_dict_str_num.items():
    bytes_key = key.encode("utf-8")
    tuple_bytes_key = tuple(bytes([x]) for x in bytes_key)
    output_dict[tuple_bytes_key] = value

# print(output_dict)

output_dict = {
 (b'l', b'o', b'w'): 5,
 (b'l', b'o', b'w', b'e', b'r'): 2,
 (b'w', b'i', b'd', b'e', b's', b't'): 3,
 (b'n', b'e', b'w', b'e', b's', b't'): 6
 }

updated_dict = {}
pair2seq_dict:dict[
                  tuple[bytes,bytes],
                  set[tuple[bytes,...]
                                    ]] = defaultdict(set)
for key,value in output_dict.items():
    # key  (b'l', b'o', b'w')
    # print(f'key = {key}')
    # print(f'length of key ={len(key)}')
    # print(value)
    key_len = len(key)
    for x in range(key_len -1):
        # print(f'{key[x]}+ {key[x+1]}')
        '''
        应为此处是tuple 所以 key[x] 是bytes 而不是 整数
        '''
        new_tuple = (key[x],key[x+1])
        updated_dict[new_tuple] = updated_dict.get(new_tuple,0) + value
        pair2seq_dict[new_tuple].add(key)
# print(updated_dict)
print(pair2seq_dict)