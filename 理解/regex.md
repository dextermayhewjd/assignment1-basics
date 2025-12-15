# 浅析LLM里的 tokenizer部分的 regex
```python
>>> import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
>>> re.findall(PAT, "some text that i'll pre-tokenize")

['some', ' text', ' that', ' i', "'ll", ' pre', '-', 'tokenize']
```
目录
1. 这个表达的是什么 [跳转](#这个表达的是什么)
2. 为什么工程里使用的是re.finditer 而不是re.findall [跳转](#为什么工程里使用的是refinditer-而不是refindall)
3. 这里英文和中文会有什么区别[跳转](#这里英文和中文会有什么区别)
4. 如何把正则表达式 1.正确 2. 保持可读性[跳转](#如何把正则表达式-1.正确-2.-保持可读性)

## 这个表达的是什么

### 1️⃣ `'(?:[sdmt]|ll|ve|re)`
1. 匹配什么？

英语里的缩写后半部分：  
`'s   'd   'm   't   'll   've   're`  
例如：  
`I'll   →  I + 'll
don't →  don + 't`

2. 如果不这样做：  
'll 会被拆成 ' + ll
或者 ' 被当成标点

👉 GPT-2 希望：  
'll 这种是一个稳定的 subword
因为它语义和语法都很固定

### 2️⃣ `?\p{L}+`
1. 匹配什么？  
字母序列（Unicode Letter）

包括：
英文
中文
日文
其他语言字母

前面的 ? 非常关键 👇

2. ? 是什么意思？  
可选的一个空格
也就是说：  
" text" 会整体匹配成 " text"  
 
3. 为什么要把空格“黏”到词上？
这是 GPT-2 的一个非常重要设计：  
不把空格当成独立 token  
而是把它作为“词的属性”  

比如：

`"hello world"
→ ["hello", " world"]`  
这样：  
1. "world" 和 " world" 是不同 token  
2. 模型能学到“词在句首 vs 句中”的区别  

### 3️⃣ `?\p{N}+`
1. 匹配什么？ 
数字序列

例如：
" 123", "42", "2025"

为什么数字要单独处理？

1. 数字有很强的内部结构
2. 不希望 2025 被打碎成 2 0 2 5
3. 统计上更稳定

### 4️⃣ `?[^\s\p{L}\p{N}]+`
1. 匹配什么？  
非空白、非字母、非数字的连续字符  
也就是：
标点
符号
emoji（很多）  

例如：    
"!!!"
"..."
"😊"
"-"
同样：

前面也允许一个空格 ?

所以 " -" 会作为整体

### 5️⃣ `\s+(?!\S)`
这是最“regex”的一条
分解来看：

\s+：多个空白

(?!\S)：后面不能跟非空白

👉 含义：

行尾 / 段尾的空白

例如：

"hello   \n"


这条规则是为了：

不丢掉尾部空白

保证编码/解码完全可逆

### `6️⃣ \s+`
兜底规则

任何剩下的空白

如果前面都没匹配上，用它。

四、把整个 regex 用一句“人话”总结

你现在可以这样理解这个 pre-tokenizer：

它会把文本切成「带空格的词 / 数字 / 标点 / 缩写 / 空白」这些稳定的小块，使得 BPE 在这些块内部学习 byte 合并时，既高效，又不会跨越明显的语言边界。



## re.finditer 而不是re.findall的使用区别 
一、re.findall vs re.finditer 使用说明（权威版）

你现在用的是 regex 包：

`import regex as re`


⚠️ 注意：下面的说明 适用于 regex，不是 stdlib 的 re，但行为在这里是一致的。

`1️⃣ re.findall
用法
re.findall(pattern, string)`

行为

返回一个 list

list 中的每一项是：

没有捕获组 → 整个 match（str）

有捕获组 → tuple / str（只返回组）

示例（你原来能跑通的）
re.findall(PAT, "some text")


返回类似：

['some', ' text']

`2️⃣ re.finditer
用法
re.finditer(pattern, string)`

行为

返回一个 iterator

每次迭代得到一个 Match object

必须调用：

m.group()


才能拿到匹配的字符串

三、立刻验证（请你直接复制运行）
① 打印 Match 本身（不是 group）
for m in re.finditer(PAT, english_words):
    print(m)


如果你看到类似：

<regex.Match object; span=(0, 4), match=''>


那问题就被我们 彻底确认 了。

四、100% 正确、不会踩坑的写法（强烈推荐）
✅ 明确取 group(0)
for m in re.finditer(PAT, english_words):
    print(repr(m.group(0)))


而不是：

m.group()

五、如果你想「行为完全等价于 findall」

这是官方推荐对等写法：

[m.group(0) for m in re.finditer(PAT, text)]

## 为什么工程里使用的是re.finditer 而不是re.findall
1️⃣ re.findall：是的，一次性生成 list（全进内存）
行为（准确版）
re.findall(pattern, text)


会 完整扫描 text

把 所有匹配结果 放进一个 list

list 里的元素是：

str（或 tuple，如果有捕获组）

👉 所有结果都会同时常驻内存

内存模型（直观）
text  ──扫描──▶  ['some', ' text', ' that', ...]


内存占用：O(#matches)

对 tokenizer 训练（大语料）不友好

2️⃣ re.finditer：是“惰性生成”，但不是“引用对象”

你说的这一句需要纠正一下：

❌ “找的是 match object 的 reference”

正确说法是：

✅ finditer 返回一个迭代器，每次迭代会生成一个新的 Match 对象

具体行为（准确）
it = re.finditer(pattern, text)


it 是一个 iterator

每 next(it) 一次：

regex 引擎 继续扫描

构造一个 新的 Match 对象

返回给你

这个 Match 对象：

包含：

匹配的 span（起止位置）

匹配到的字符串（或组）

不是指向某个“共享对象”的引用

内存模型（对比 findall）
findall
[text] → scan → [str, str, str, ...]   # 全部存住

finditer
[text] → scan → Match → 处理 → 丢弃
                      ↓
                    下一个 Match


内存占用：≈ O(1)

只要你不自己存 Match，就不会积累

3️⃣ Match 对象到底存了什么？

一个 Match 对象里有（简化）：

m.group(0) → 匹配到的字符串（新创建的 str）

m.start(), m.end() → 索引

m.groups() → 捕获组内容

⚠️ 注意：

group(0) 不是指向原字符串的 view

是一个新的 Python str 对象

但只在你使用时存在

4️⃣ 为什么 tokenizer / CS336 强调用 finditer？

因为 tokenizer 的训练流程是：

for doc in corpus:
    for m in re.finditer(PAT, doc):
        counter[m.group(0)] += 1


特点：

不保存中间 token

一边扫描

一边统计

适用于 TB 级文本

5️⃣ 用一句“完全准确、不误导”的总结

你可以这样记（或这样写在作业里）：

re.findall eagerly computes all matches and stores them in a list, which can be memory-intensive.
re.finditer returns an iterator that lazily yields Match objects one by one, allowing streaming processing with constant memory usage.

6️⃣ 你现在的理解水平评估（实话）

你现在已经：

✅ 分清了 eager vs lazy

✅ 知道 tokenizer 关心的是 统计而不是打印

🔧 只差把这个模式用到 BPE merge 统计里
## 这里英文和中文会有什么区别

https://chatgpt.com/g/g-p-693f75d2365c8191baf9aaa7038e3595-cs336xiao-xi-jie/c/693f9fe9-b124-832b-8f73-6bbae43ea2ea  
一、这个 PAT 是“为谁设计的”？

你给的正则：

PAT = r"""'(?:[sdmt]|ll|ve|re)
        | ?\p{L}+
        | ?\p{N}+
        | ?[^\s\p{L}\p{N}]+
        |\s+(?!\S)
        |\s+"""


👉 这是 GPT-2 / GPT-3 系 tokenizer 的经典 pre-tokenizer 模板

设计假设是：

\p{L}+：
👉 “字母是连续出现才有意义”

'll, 've：
👉 英文缩写

?xxx：
👉 把空格编码进 token

核心目标：最大化英文文本的压缩效率

二、它对中文哪里“不合适”？
🔴 问题 1：\p{L}+ 会把多个汉字合并

在 Unicode 中：

中 国 文


每一个汉字：

都属于 \p{L}（Letter）

所以 \p{L}+ 会匹配：

"中文"
"自然语言处理"


📉 结果：

pre-tokenizer 把一整串汉字当成一个 unit
❌ 这违背了“中文以字为基础”的统计假设

🔴 问题 2：+ 破坏了 BPE / Unigram 的学习前提

BPE 期望看到的是：

[自][然][语][言][处][理]


而不是：

[自然语言处理]


否则：

高频子结构被“藏”在 token 内

子词合并空间被极大压缩

中文建模效果明显下降

🔴 问题 3：英文缩写规则对中文无意义
'(?:[sdmt]|ll|ve|re)


只对：

I'm

I've

he'll

中文完全无用

还增加正则复杂度

三、所以：这个 PAT 对中文会“看起来能跑，但语义是错的”

你如果现在测试：

re.findall(PAT, "我喜欢自然语言处理")


得到的很可能是：

["我喜欢自然语言处理"]


👉 这是 pre-tokenizer 的灾难级结果
不是报错，而是静默地做错了事

四、中文正确的 pre-tokenizer 应该长什么样？
最小正确版本（教学 / cs336 级别）
PAT_ZH = r"""
    \p{Script=Han}        # 单个汉字
  | [A-Za-z]+             # 英文连续
  | \p{N}+                # 数字
  | [^\s\p{Script=Han}A-Za-z\p{N}]+
  | \s+
"""


👉 核心变化：

❌ 移除 \p{L}+

✅ 显式区分 Han 和 Latin

✅ 汉字 不加 +

示例效果
我love NLP2024！


↓

[我][love][ ][NLP][2024][！]

五、如果你一定要“兼容英文 + 中文”，正确思路是什么？
❌ 错误思路

“用一个 regex 同时 cover 所有语言”

✅ 正确思路（主流模型做法）

按 Unicode Script 分派规则

伪代码：

if char in Han:
    emit char
elif char in Latin:
    buffer += char
elif char is digit:
    buffer += char
else:
    flush buffer
    emit char


👉 这才是 tokenizer 工程的本质

regex 只是实现方式，不是核心思想。

六、给你一个可以直接写进作业的总结句

GPT-2 风格的 pre-tokenizer 使用 \p{L}+ 假设字母语言的语义单位由连续字符组成，
这一假设不适用于汉字体系。
对中文文本，pre-tokenization 应在 Unicode Script 层面将每个汉字视为独立单元，
由后续的 BPE / Unigram 学习高频多字结构。
## 如何把正则表达式 1.正确 2. 保持可读性



 