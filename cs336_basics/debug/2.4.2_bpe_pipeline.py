'''
总的逻辑
1. 把每个词都按照PAT 先分词分开来 
使用PAT

2. 使用统计频率 统计每个词的频率
low:5

3. 将词 encode 成 bytes后 
b'low':5

4.统计 词的内部pair
(b'l',b'o'):5

5. 找到内部最高的 一对pair 
做 merge
    1.在vocab: 
        key:创建的 新的index 
        value: merge的 新的 长度为n的bytes对象
    
    2.记录规则  
        
    3.在原先的词中 找到这一对
        替换为新的 bytes 对象 

6. 重点考虑的是 长度为n的bytes对象 会不会对原来的算法有影响

7 如果没有影响 就只要重复4-5就可以了对吧
'''

'''
task 1 统计到底有多少个词
1. 使用字典
2. 用空格分开text生成list
然后java hashmap经典统计频率 来生成counts
'''

text = "low low low low low lower lower widest widest widest newest newest newest newest newest newest"
counts = {}

text_list = text.split(' ')
for t in text_list:
    counts[t] = counts.get(t, 0) + 1

# print(counts)


'''
task 2 
创建byte的字典 
# '''
vocab : dict[int,bytes] = { x:bytes([x]) for x in range(256) }
# print(vocab)
max_index = 256
# print('==================上面是vocab===================================')
'''
task3 
把counts 里的 dict[str,int]

变成 word_bytes_counts = {
   (b'l', b'o', b'w'): 5,
   (b'l', b'o', b'w', b'e', b'r'): 2,
   ...
}
意思是把 str 变成 tuple(b'x',b'y',b'c')

1. str 变成b'str' 容易
    s.encode("utf-8")

2. b'str'变成tuple(b's',b't',b'r') 比较麻烦

    2.1生成tuple 的话 
        用生成器表达式 generator expression + tuple 
        tuple(expression(x) for x in something)

    2.2如何切分expression中 单独切分出一个个的 b'x'

        2.2.1 bytes是字节的序列可遍历 
            list(b)
            #[108,111,119]
            
        2.2.2 但是遍历获得的x是 int整型
            b = b"ABC"
            b[0]      # 65(int)
            
        2.2.3 把int变成对应的bytes   
            bytes([65]) # b'A'
            
            可以bytes([x]) 因为遍历 得到的是int 

test 案例是把 ABC 变成 tuple (b'a',b'b',b'c') 
'''

# example = 'abc'
# example_bytes = example.encode("utf-8")

# tuple_bytes_example = tuple(bytes([x]) for x in example_bytes)

# print(tuple_bytes_example)

# {'low': 5, 'lower': 2, 'widest': 3, 'newest': 6}
test_dict_str_num = {'low': 5, 'lower': 2, 'widest': 3, 'newest': 6}

output_dict = {}
for key,value in test_dict_str_num.items():
    bytes_key = key.encode("utf-8")
    tuple_bytes_key = tuple(bytes([x]) for x in bytes_key)
    output_dict[tuple_bytes_key] = value

# print(output_dict)
'''
{
  (b'l', b'o', b'w'): 5, 
  (b'l', b'o', b'w', b'e', b'r'): 2, 
  (b'w', b'i', b'd', b'e', b's', b't'): 3, 
  (b'n', b'e', b'w', b'e', b's', b't'): 6
}
'''


'''
task4
把上面 dict[tuple[bytes,...], int]  
1. 提取每一个键值对
eg. (b'l', b'o', b'w'): 5

2.组合 在新的键 b'lo' b'ow'
    dict[tuple[bytes,bytes], int]

3. 每个都是+=5

4. 下一个键值对

return dict[tuple[bytes(单个吧这会),bytes(单个吧这会)],int]
'''
updated_dict = {}
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

# print(updated_dict)

'''
目前要统计出哪个key value的值最高
这里要熟悉 sorted的用法 返回的是一个list

'''

sorted_dict = sorted(updated_dict.items(),key=lambda x: (x[1],x[0]),reverse= True) 
# print(sorted_dict)
max_value_key = sorted_dict[0][0]
# print(f'要加入的一对bytes{max_value_key}')

'''
开始融合
思路是
记住哪一组是最高的
把这两个byte 组成一块

就在vocab中加入这个词
vocab: dict[int:bytes]
所以得统计目前最大的是哪一个 


遍历一下所有token的bytes版本
如果是这个最高的

'''
#merged 可以直接bytes拼接 
print(max_value_key[0]+max_value_key[1])

vocab[max_index] = max_value_key[0] + max_value_key[1]
# print(vocab) 
# 对了



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
  
# token1 = (b'w', b'i', b's', b't', b'd', b'e')
# pair1 = (b's', b't')
# new_token1 = b'st'
# test_output = (b'w', b'i', b'd', b'e', b'st')  
# output = merge_pair_in_token(token=token1,pair=pair1)
# print(output)


