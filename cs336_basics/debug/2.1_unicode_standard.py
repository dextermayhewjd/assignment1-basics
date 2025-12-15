
x = chr(0)

y = ord('牛') # 将 str 变成 code point 

z = ord('s')
print(f'print x = {x}, repr x is {repr(chr(0))}')
# repr(chr(0))    # 返回 "'\\x00'" 
print(f'print y = {y}, reverse y = {chr(y)}')

print(f'print z = {z}, reverse z = {chr(z)}')

chr(0)
print(chr(0))
print("this is a test" + chr(0) + "string")

# ord()  是字符变成 int
# ord（c:str）-> int


