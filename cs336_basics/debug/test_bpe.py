'''
test_corpus.txt 是这样子的 

low low low low low
lower lower widest widest widest
newest newest newest newest newest newest
'''
# f = open("test_corpus.txt", "rt")
# print(f)
# print(f.read())

#这个 自动开关
# with open("test_corpus.txt") as f:
#   print(f.read())

# 不是很明白这个的问题
with open("test_corpus.txt") as f:
  for x in f:
    print(x)
'''
首先呢 我们的vocabulary 要有
1. 256byte的 初始值 
2. special token <|endoftext|> 
'''

'''
如何创建byte是
1️⃣ 单个 byte（最关键）
bytes([97])   # b'a'
bytes([0])    # b'\x00'
bytes([255])  # b'\xff'

📌 规则：
bytes() 接收的是 0–255 的整数序列
每个整数 → 一个 byte
'''

vocab = []
for i in range(256):
  vocab.append(bytes([i]))
vocab.append(b'<|endoftext|>')
print(vocab)

assert len(vocab) == 257