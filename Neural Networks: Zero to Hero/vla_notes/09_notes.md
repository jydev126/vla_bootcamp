# 09 完整笔记｜GPT Tokenizer：BPE、特殊 token、接口设计和动作离散化

## 0. 这节课到底要学会什么

Tokenizer 不是无聊预处理。它是模型和世界之间的接口。

核心问题：

```text
原始文本如何变成模型能处理的整数序列？
```

对 VLA 来说，这个问题会扩展成：

```text
图像如何变成 visual tokens？
动作如何变成 action tokens？
地图/状态/轨迹如何变成 token 或 embedding？
```

所以这节不只是 NLP。它是“接口设计课”。

## 1. 为什么不能直接输入字符串

神经网络处理的是数字 Tensor。

文本必须变成：

```text
字符串 -> token pieces -> token ids -> embeddings
```

模型真正看到的是 token id 序列，再查 embedding table。

## 2. 字符级 tokenizer

最简单：每个字符一个 token。

优点：

```text
简单
不会有未知词
decode 容易
```

缺点：

```text
序列太长
计算成本高
不能高效表达常见词/片段
```

## 3. 词级 tokenizer

每个词一个 token。

优点：

```text
序列短
语义片段大
```

缺点：

```text
词表巨大
新词/OOV 麻烦
拼写变化麻烦
多语言麻烦
```

所以现代 LLM 多用 subword tokenizer。

## 4. BPE 的核心思想

BPE = Byte Pair Encoding。

训练过程：

```text
1. 从基础单位开始，比如 byte/char
2. 统计相邻 pair 出现频率
3. 合并最高频 pair，形成新 token
4. 重复很多次，直到词表达到目标大小
```

例子：

```text
t + h -> th
th + e -> the
```

常见片段会成为单 token，不常见片段会被拆得更细。

## 5. BPE 为什么有效

它折中字符级和词级：

```text
常见词/片段: token 少，效率高
罕见词: 可以拆成子词/字节，不会完全 OOV
```

这让模型在效率和泛化之间取得平衡。

## 6. encode

encode 做：

```text
raw string
-> bytes / unicode handling
-> regex pre-tokenization
-> BPE merges
-> token ids
```

注意真实 tokenizer 不只是 BPE merge，还有正则切分、空格处理、特殊 token 处理。

## 7. decode

decode 做反向：

```text
token ids -> token bytes/pieces -> string
```

但 decode 不总是符合人的直觉。

例如空格可能被并入 token：

```text
" hello" 可能是一个 token
"hello" 是另一个 token
```

这会造成很多奇怪现象。

## 8. Unicode 和 bytes

文本不是只有英文字符。

Unicode 字符可能由多个 bytes 组成。

byte-level BPE 的好处：

```text
任何字符串都能表示
不会出现真正无法编码的字符
```

代价：某些语言或符号 tokenization 效率低。

## 9. tokenization 的隐藏影响

Tokenizer 会影响模型能力：

```text
数字拆分方式影响算术
空格处理影响格式
大小写影响 token
多语言 token 密度不同
代码缩进/符号影响代码能力
罕见词拆分影响记忆和泛化
```

很多 LLM 的“奇怪行为”可以追到 tokenizer。

## 10. special tokens

特殊 token 是协议，不是普通文本。

例如：

```text
<|endoftext|>
<bos>
<eos>
<pad>
<image>
<latent>
<action_bos>
```

它们告诉模型结构信息：

```text
序列开始/结束
角色分隔
模态占位
动作开始
latent 位置
```

特殊 token 是否被正确加入 tokenizer 和 embedding table，是工程关键。

## 11. vocab size

词表越大：

```text
单段文本 token 数可能减少
embedding/lm head 参数增加
稀有 token 学得可能不充分
```

词表越小：

```text
序列变长
上下文窗口压力变大
计算更贵
```

VLA 里的 action bins 也有类似 tradeoff。

## 12. tokenizer 训练数据的偏置

BPE merge 来自 tokenizer 训练语料。

如果语料里某些字符串高频，它们更可能成为 token。

所以 tokenizer 本身带数据偏置。

对 VLA 来说，如果动作离散 bins 或轨迹 token 来自数据统计，也会带驾驶数据分布偏置。

## 13. GPT-2 tokenizer 和 GPT-4 tokenizer 差异

课程会展示不同 tokenizer 对同一句话切分不同。

这说明：

```text
模型和 tokenizer 是绑定的
不能随便换 tokenizer
```

换 tokenizer 意味着 embedding table、LM head、训练分布都变了。

## 14. ChatML / 对话模板

Chat 模型通常不只是普通文本，而有对话模板：

```text
<system> ...
<user> ...
<assistant> ...
```

这些角色标记会被 tokenizer 编码成特殊 token 或普通 token 序列。

VLA 也可能有结构化 prompt：

```text
Command:
Ego state:
History trajectory:
Scene description:
Predict future trajectory:
```

prompt 格式会影响模型理解。

## 15. tokenizer 和上下文长度

同样的内容，不同 tokenizer 产生 token 数不同。

这影响：

```text
能放进上下文的信息量
推理延迟
attention 计算量
成本
```

VLA 中视觉 token 本来就多，文本/状态格式如果冗长，会进一步挤占上下文。

## 16. action tokenizer

把连续动作变成离散 token：

```text
continuous value -> bin index -> action token id
```

例如 steering 范围：

```text
[-1, 1]
```

分成 256 个 bins。每个连续值映射到一个 bin。

## 17. action tokenization 的代价

离散化会损失精度：

```text
0.101 和 0.104 可能同一个 bin
0.101 和 0.109 可能不同 bin
```

bin 太少：动作粗糙。

bin 太多：分类更难，数据更稀疏，词表变大。

这是 VLA action-token 路线必须权衡的点。

## 18. 多维 action 如何 token 化

一个轨迹可能是：

```text
6 个 waypoints，每个 waypoint 有 x,y
=> 12 个连续数
```

可以变成：

```text
12 个 action token ids
```

也可以用更结构化的 token：

```text
x-bin token
y-bin token
waypoint separator
action step token
```

设计会影响模型学习难度。

## 19. visual tokenization

VLM 中：

```text
image -> patches -> vision encoder -> visual tokens
```

这也是广义 tokenizer：把连续图像变成有限长度 token sequence。

关键 tradeoff：

```text
patch 越小，token 越多，细节越多，计算越贵
patch 越大，token 越少，细节损失更多
```

## 20. latent/special token 路线

VLA 可能插入特殊 token 或 latent slots：

```text
<latent_1> <latent_2> ...
```

但字符串本身没意义。真正有意义的是：

```text
这些位置经过 Transformer 后的 hidden state
```

训练会迫使这些 hidden state 承载规划、语义、世界状态或风险信息。

## 21. tokenizer 是接口契约

一旦模型训练好了，tokenizer 就是契约。

不能随便改变：

```text
vocab
merge rules
special token ids
chat template
action bin 定义
坐标归一化方式
```

否则模型输入输出语义会错位。

## 22. 对完整 VLA 能力栈的意义

VLA 里至少有几种“tokenization”：

```text
文本 tokenization
图像 patch/tokenization
状态/history embedding
地图元素 tokenization
action discretization
latent/special token protocol
```

每一种都决定信息瓶颈。

你要问：

```text
信息损失在哪里？
序列长度是多少？
token 粒度是否适合任务？
特殊 token 是否被训练过？
动作离散化误差是否可接受？
```

## 23. 复习自测

1. 字符级 tokenizer 和 BPE 的优缺点是什么？
2. BPE merge 是怎么训练出来的？
3. 为什么 tokenizer 会影响模型能力？
4. special token 为什么是协议？
5. 为什么模型和 tokenizer 不能随便拆开？
6. action tokenizer 和文本 tokenizer 有什么共同点？
7. action bins 太多/太少分别有什么问题？
8. visual tokens 为什么也是广义 tokenization？
9. latent token 字符串本身为什么没意义？
10. VLA 中哪些接口一旦训练好就不能随便改？
