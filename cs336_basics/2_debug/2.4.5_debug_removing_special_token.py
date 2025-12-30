import re


text = "Hello world<|endoftext|>Second doc<|endoftext|>Third doc"
special_tokens = ["<|endoftext|>"]

def split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    if not special_tokens:
        return [text]
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    return re.split(pattern, text)

# 不用正则 
text = "A<X>B<X>C"
print(text.split("<X>"))

# 要用正则 来匹配多种 special token 
# 因为 spilit 不支持 pipeline 
text = "AAAAAAAA<|X>B<Y>C"
special_tokens = ["<|", ">","<"]

pattern = "|".join(tok for tok in special_tokens)
chunks = re.split(pattern, text)
print(chunks)

pattern = "|".join(re.escape(tok) for tok in special_tokens)
chunks = re.split(pattern, text)
print(chunks)