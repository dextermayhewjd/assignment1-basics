# requires `regex` package
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# english_words = "some text that i'll pre-tokenize"
# print(f"The original sentence is '{english_words}'")
# print(re.findall(PAT, english_words))


# chinese_words = "我倒要看看他分不分的出来，傻子gpt"
# print(f"The original sentence is '{chinese_words}'")
# print(re.findall(PAT, chinese_words))


# import regex as re

PAT2 = r"""
'(?:[sdmt]|ll|ve|re)
| \ ?\p{L}+
| \ ?\p{N}+
| \ ?[^\s\p{L}\p{N}]+
| \s+(?!\S)
| \s+
"""

english_words = "some text that i'll pre-tokenize"
print(f"The original sentence using PAT '{english_words}'")
for m in re.finditer(PAT, english_words):
    print(m)
    print(m.group())

print([m.group(0) for m in re.finditer(PAT, english_words)])
# print(f"The original sentence using PAT2 '{english_words}'")
# for m in re.finditer(PAT2, english_words, flags=re.VERBOSE):
#     print(m)


# chinese_words = "我倒要看看他分不分的出来，傻子gpt"
# print(f"The original sentence is '{chinese_words}'")
# for m in re.finditer(PAT, chinese_words):
#     print(m)