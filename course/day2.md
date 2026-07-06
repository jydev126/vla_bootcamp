# Day 2：Transformer / Attention / Causal LM 上课台词

## 开场：今天到底要学会什么

好，我们今天进入 Day 2。

昨天我们已经完成了 PyTorch 最小闭环，也就是知道了什么是 Tensor，什么是 `nn.Module`，什么是 `forward`，什么是 loss，什么是 `backward`，什么是 `optimizer.step()`。

但是昨天的模型还不是语言模型。

今天我们要开始接触大模型最核心的一条主线：Transformer，Attention，Causal Language Model。

你先不要被这些词吓到。

今天我们的目标不是训练一个真正的大模型，也不是读懂所有论文，而是手写一个最小 GPT。

所谓最小 GPT，就是从一段字符文本开始，最后让模型能够生成类似训练文本风格的字符。

今天必须吃透这条链路：

文本，先经过 tokenizer，变成 input_ids。

input_ids 经过 embedding，变成向量。

向量进入 self-attention，让每个 token 能看到上下文。

然后经过 Transformer block。

最后输出 logits。

logits 和真实的下一个 token 做 cross entropy。

训练目标就是 next-token prediction，也就是根据前面的 token 预测下一个 token。

所以你今天要记住一句话：

GPT 的本质不是神秘的智能系统。

GPT 的基本训练方式就是：给它一段 token 序列，让它不断预测下一个 token。

---

## 第一部分：Tokenizer 到底是什么

我们先讲第一个文件：

`01_char_tokenizer.py`

今天我们用的是字符级 tokenizer。

也就是说，一个字符就是一个 token。

比如这句话：

`the car slows down and the ego vehicle should brake`

在字符级 tokenizer 里面，`t` 是一个 token，`h` 是一个 token，`e` 是一个 token，空格也是一个 token。

注意，空格也是字符，所以空格也是 token。

这点非常重要。

很多初学者会以为 tokenizer 只处理单词，其实不是。

token 可以是字符，可以是子词，也可以是词。

真实大模型一般不是字符级 tokenizer，而是 BPE 或者 SentencePiece 这种子词 tokenizer。

但是今天我们不碰那些复杂东西。

今天只用字符级 tokenizer，因为它最直观。

---

### 1.1 原始文本不能直接进模型

现在你要先建立一个基本认识：

模型不能直接处理字符串。

神经网络里面流动的是数字张量，不是中文，不是英文，也不是字符串。

所以这句话：

`the car slows down`

不能直接喂给模型。

我们必须先把它变成整数。

比如：

`the`

可能会变成：

`[17, 8, 5]`

这里的 17、8、5 不代表真正的语义。

它们只是字符表里面的编号。

所以我们说：

tokenizer 的作用，就是把文本里的离散符号映射成整数 id。

这个整数 id 就叫 token id。

一整串 token id，就叫 input_ids。

---

### 1.2 看这几行代码

我们看代码：

```python
text = "the car slows down and the ego vehicle should brake"
```

这就是我们的原始文本。

然后：

```python
chars = sorted(list(set(text)))
```

这一行看起来很简单，但是很关键。

我们拆开讲。

`set(text)` 的意思是，把文本中出现过的所有字符去重。

比如文本里出现了很多次 `e`，但是 `set` 之后只保留一个 `e`。

然后 `list(...)` 是把集合变成列表。

最后 `sorted(...)` 是排序，让字符表顺序稳定。

为什么要排序？

因为如果不排序，每次运行时字符的顺序可能不一样。

字符顺序不一样，字符对应的 id 就不一样。

这样今天运行一次，`t` 是 17，明天运行一次，`t` 可能变成 3，调试起来会很乱。

所以我们用 `sorted` 固定顺序。

然后：

```python
vocab_size = len(chars)
```

`vocab_size` 就是词表大小。

注意，这里虽然叫词表，但是我们是字符级 tokenizer，所以它其实是字符表大小。

如果文本里一共出现了 20 种不同字符，那么 `vocab_size` 就是 20。

---

### 1.3 stoi 和 itos

继续看：

```python
stoi = {ch: i for i, ch in enumerate(chars)}
```

`stoi` 的意思是 string to integer。

也就是字符到整数。

比如：

```python
{
  'a': 1,
  'b': 2,
  'c': 3
}
```

当然实际字符表里还会有空格、换行、其他字母。

然后：

```python
itos = {i: ch for ch, i in stoi.items()}
```

`itos` 的意思是 integer to string。

也就是整数到字符。

为什么需要两个表？

因为训练时，我们需要从文本变成 id。

生成时，模型输出的是 id，我们又需要把 id 还原成文本。

所以：

`encode` 用 `stoi`。

`decode` 用 `itos`。

---

### 1.4 encode 和 decode

我们看：

```python
def encode(s: str) -> list[int]:
    return [stoi[ch] for ch in s]
```

这句话的意思是：

给我一个字符串，我遍历里面的每个字符，然后用 `stoi[ch]` 找到它的编号。

比如：

`"the"`

会被转换成：

`[id_t, id_h, id_e]`

再看：

```python
def decode(ids: list[int]) -> str:
    return "".join([itos[i] for i in ids])
```

这句话的意思是：

给我一串整数 id，我用 `itos[i]` 找回每个字符，然后用 `"".join(...)` 拼成字符串。

所以：

`decode(encode(text))`

应该等于原始的 `text`。

这是 tokenizer 最基本的验收标准。

---

### 1.5 这里你要背什么

讲到这里，你要背下来：

tokenizer 不是模型。

tokenizer 不负责理解语言。

tokenizer 只是一个映射工具。

它把文本变成 input_ids，也能把 input_ids 变回文本。

模型真正看到的不是字符串，而是 input_ids。

input_ids 是整数序列。

---

## 第二部分：Bigram LM，最小语言模型

现在进入第二个文件：

`02_bigram_lm.py`

这个文件实现的是 Bigram Language Model。

Bigram 的意思是两个 token 的关系。

简单说，它只根据当前 token 预测下一个 token。

比如文本是：

`the car`

我们切成字符后：

`t h e 空格 c a r`

训练时可以变成：

输入：

`t h e 空格 c a`

目标：

`h e 空格 c a r`

也就是说：

看到 `t`，预测 `h`。

看到 `h`，预测 `e`。

看到 `e`，预测空格。

看到空格，预测 `c`。

这就是 next-token prediction。

---

### 2.1 先看 import

文件开头：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
```

这里我们讲一下。

`torch` 是 PyTorch 主库，里面有 Tensor、随机数、矩阵运算、保存加载等功能。

`torch.nn` 是神经网络模块库。

比如 `nn.Module`，`nn.Linear`，`nn.Embedding`，`nn.LayerNorm` 都在这里。

我们通常写：

```python
import torch.nn as nn
```

这样后面就能写 `nn.Module`，而不是每次写 `torch.nn.Module`。

然后：

```python
import torch.nn.functional as F
```

`F` 里面放的是一些函数形式的操作。

比如：

`F.cross_entropy`

`F.softmax`

这些不是有状态的层，而是直接调用的函数。

简单理解：

`nn.Linear` 是一个带参数的层。

`F.softmax` 是一个计算函数。

---

### 2.2 torch.manual_seed(42)

接下来看：

```python
torch.manual_seed(42)
```

这行代码特别容易被忽略，但是一个好工程师不能忽略它。

它的作用是固定随机种子。

什么叫随机种子？

神经网络里有很多随机行为。

比如模型参数初始化是随机的。

比如随机采样 batch 是随机的。

比如生成文本时采样下一个 token 也是随机的。

如果不固定随机种子，你每次运行代码，结果都可能不一样。

这会导致你调试很痛苦。

比如你今天 loss 下降，明天 loss 不下降，你不知道是代码变了，还是随机性导致的。

所以在学习和调试阶段，我们经常写：

```python
torch.manual_seed(42)
```

42 不是特殊数字，只是大家常用的一个数字。

你也可以写 0，1，1234。

重点不是 42，重点是固定随机性，让实验尽量可复现。

但是你要知道，它不能保证所有 GPU 操作都完全确定。

今天我们不用管那么深。

你只要记住：

`torch.manual_seed(42)` 是为了让随机初始化和随机采样尽量稳定，方便调试和复现实验。

---

### 2.3 构造训练文本

接下来：

```python
text = """
the car slows down and the ego vehicle should brake
...
""" * 50
```

这里我们把几句话重复 50 次。

为什么要重复？

因为文本太短，模型没什么可学。

我们今天不是追求泛化能力，只是要看到 loss 下降，看到 generate 能生成类似风格的字符。

所以重复文本是可以的。

它的作用是给模型足够多的训练样本。

当然，真实大模型不会这样训练。

真实大模型会用海量真实文本。

但今天我们是教学实验，先保证闭环跑通。

---

### 2.4 data 是什么

然后：

```python
data = torch.tensor(encode(text), dtype=torch.long)
```

这行代码非常重要。

`encode(text)` 得到的是 Python list。

比如：

`[1, 2, 3, 4, 5]`

但是模型不能直接训练 Python list。

我们要把它转成 PyTorch Tensor。

所以用：

```python
torch.tensor(...)
```

这里 `dtype=torch.long` 也很关键。

为什么不是 float？

因为 input_ids 是索引。

索引必须是整数类型。

`nn.Embedding` 的输入要求是整数 id，一般就是 `torch.long`。

你不能把 token id 写成 float。

比如 token id 是 3，它表示查 embedding 表第 3 行。

这不是连续数值，不应该是 3.0。

所以这里用 `torch.long`。

你要背：

input_ids 是整数索引，所以 dtype 要用 `torch.long`。

---

### 2.5 get_batch：语言模型训练样本怎么来

我们看：

```python
def get_batch():
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)
```

这段非常重要。

我们一点点拆。

`batch_size` 是一次训练取多少条样本。

`block_size` 是每条样本的长度，也就是序列长度。

如果：

```python
batch_size = 16
block_size = 16
```

那 `x` 的 shape 就是：

`[16, 16]`

意思是：

16 条样本，每条样本 16 个 token。

然后：

```python
ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))
```

这行是在随机选起点。

比如随机选到位置 100，那么：

```python
x = data[100:116]
y = data[101:117]
```

你看，`y` 比 `x` 向右移动了一位。

这就是 next-token prediction 的训练构造。

x 是输入。

y 是目标。

模型看到 x 的每个位置，要预测 y 的对应位置。

所以：

x 第 0 个 token 的目标，是 y 第 0 个 token，也就是原文本里的下一个 token。

这就是语言模型训练的核心。

---

### 2.6 nn.Embedding 在 Bigram 里做了什么

现在看模型：

```python
class BigramLanguageModel(nn.Module):
```

只要你写模型，基本都要继承 `nn.Module`。

继承 `nn.Module` 的好处是：

PyTorch 能自动管理你的参数。

比如后面调用：

```python
model.parameters()
```

PyTorch 就能知道模型里有哪些参数需要训练。

然后看：

```python
self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
```

这一行非常重要。

`nn.Embedding` 本质是一张表。

输入是 token id。

输出是对应行的向量。

一般情况下，我们会写：

```python
nn.Embedding(vocab_size, n_embd)
```

意思是每个 token id 对应一个 `n_embd` 维向量。

但在 Bigram 模型里，我们写的是：

```python
nn.Embedding(vocab_size, vocab_size)
```

这就很特殊。

输入一个 token id，输出一个长度为 `vocab_size` 的向量。

这个向量直接当作 logits。

也就是说，Bigram 模型本质上是一张转移表：

当前 token 是谁，就查出“下一个 token 可能是谁”的分数。

---

### 2.7 logits 是什么

在 forward 里：

```python
logits = self.token_embedding_table(idx)
```

假设：

`idx.shape = [B, T]`

那么：

`logits.shape = [B, T, vocab_size]`

这里你一定要能解释。

`B` 是 batch size。

`T` 是序列长度。

`vocab_size` 是词表大小。

所以 `[B, T, vocab_size]` 的意思是：

每个 batch 里，每个 token 位置，都输出一个长度为 vocab_size 的预测分数。

这个预测分数就是 logits。

logits 不是概率。

logits 是未归一化分数。

比如 logits 可能是：

`[2.1, -0.3, 5.7, 0.0]`

它可以是负数。

也不要求加起来等于 1。

经过 softmax 之后，才会变成概率。

你要背：

logits 是模型对每个类别的未归一化打分。

语言模型里的 logits 是对下一个 token 的打分。

---

### 2.8 为什么 cross_entropy 前要 reshape

然后看：

```python
B, T, C = logits.shape
logits_flat = logits.reshape(B * T, C)
targets_flat = targets.reshape(B * T)
loss = F.cross_entropy(logits_flat, targets_flat)
```

为什么要 reshape？

因为 `F.cross_entropy` 通常希望输入是：

`[N, C]`

这里 `N` 是样本数量，`C` 是类别数量。

而我们现在的 logits 是：

`[B, T, C]`

也就是说，batch 和时间维是分开的。

但是对于 cross entropy 来说，每一个 token 位置都是一个分类样本。

所以我们把 B 和 T 合并。

`[B, T, C]` 变成 `[B*T, C]`

targets 从 `[B, T]` 变成 `[B*T]`

然后每一行 logits 对应一个真实 token id。

这就是语言模型 loss 的基本写法。

你要背：

训练语言模型时，每个位置都是一个分类任务。

分类类别数就是 vocab_size。

真实标签就是下一个 token id。

---

### 2.9 optimizer 训练循环

看训练代码：

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
```

`AdamW` 是一个优化器。

它负责根据梯度更新模型参数。

这里的：

```python
model.parameters()
```

就是把模型里所有可训练参数交给优化器。

然后训练循环是：

```python
logits, loss = model(xb, yb)

optimizer.zero_grad(set_to_none=True)
loss.backward()
optimizer.step()
```

这三步要背下来。

第一步：

`optimizer.zero_grad()`

清空上一轮梯度。

为什么要清空？

因为 PyTorch 默认会累积梯度。

如果你不清空，上一轮的梯度会和这一轮加在一起。

第二步：

`loss.backward()`

反向传播，计算每个参数的梯度。

第三步：

`optimizer.step()`

根据梯度更新参数。

所以训练循环的核心就是：

前向算 loss。

清空梯度。

反向传播。

更新参数。

---

### 2.10 generate 为什么只取最后一个位置

看生成代码：

```python
logits, loss = self(idx)
logits = logits[:, -1, :]
probs = F.softmax(logits, dim=-1)
idx_next = torch.multinomial(probs, num_samples=1)
idx = torch.cat((idx, idx_next), dim=1)
```

这里最重要的是：

```python
logits = logits[:, -1, :]
```

为什么只取最后一个位置？

因为生成时，我们已经有了一串 token。

比如：

`the car`

我们想生成下一个 token。

真正需要的是最后一个位置对下一个 token 的预测。

模型虽然会对所有位置都输出 logits，但生成时只需要最后一个位置。

然后：

```python
F.softmax(logits, dim=-1)
```

把 logits 变成概率。

接着：

```python
torch.multinomial(probs, num_samples=1)
```

根据概率采样一个 token。

注意，是采样，不一定每次都取最大概率。

如果每次都取最大概率，叫 greedy decoding。

这里用采样，是为了生成有一些随机性。

最后：

```python
torch.cat((idx, idx_next), dim=1)
```

把新生成的 token 拼回原序列。

然后继续下一轮。

所以生成是逐 token 进行的。

---

## 第三部分：Self-Attention

现在进入今天最核心的部分：

`03_self_attention.py`

你要先记住：

Bigram 模型的问题是，每个 token 只能根据自己预测下一个 token。

但是语言理解需要上下文。

比如：

`the car slows down because the traffic light is red`

模型预测某个词时，不能只看当前字符，它要看前面很多字符。

Self-attention 的作用就是：

让每个 token 根据上下文重新组织自己的表示。

---

### 3.1 输入 x 的 shape

我们构造：

```python
B = 2
T = 4
C = 8
H = 4
x = torch.randn(B, T, C)
```

这里的 x 是模拟 embedding 后的输入。

`B = 2`，表示 2 条样本。

`T = 4`，表示每条样本 4 个 token。

`C = 8`，表示每个 token 是 8 维向量。

所以：

`x.shape = [2, 4, 8]`

你要建立一个直觉：

Transformer 里面大多数张量都是 `[B, T, C]`

B 是 batch。

T 是时间或者序列长度。

C 是通道数，也就是 embedding dimension。

---

### 3.2 Q、K、V 是什么

Self-attention 里面，每个 token 的向量 x 会被映射成三个向量：

Q，K，V。

代码是：

```python
q = self.query(x)
k = self.key(x)
v = self.value(x)
```

这里的：

```python
self.query = nn.Linear(n_embd, head_size, bias=False)
self.key = nn.Linear(n_embd, head_size, bias=False)
self.value = nn.Linear(n_embd, head_size, bias=False)
```

都是线性层。

它们把输入的 C 维向量映射成 H 维向量。

所以：

x 是 `[B, T, C]`

q 是 `[B, T, H]`

k 是 `[B, T, H]`

v 是 `[B, T, H]`

那 Q、K、V 该怎么理解？

初学时不要背抽象定义。

你这样记：

Q 是 Query，表示：我想找什么信息。

K 是 Key，表示：我有什么信息可以被别人匹配。

V 是 Value，表示：如果别人关注我，我实际贡献什么内容。

比如当前 token 是 `car`。

它的 Q 可能在问：“我需要找和动作相关的信息。”

前面的 `slows` 的 K 可能和这个 Q 匹配度高。

于是 attention 权重就高。

最后真正被加权汇总的是 V。

---

### 3.3 为什么 attention score 是 [B, T, T]

核心代码：

```python
wei = q @ k.transpose(-2, -1) / math.sqrt(H)
```

我们只看 shape。

q 是：

`[B, T, H]`

k 是：

`[B, T, H]`

`k.transpose(-2, -1)` 之后变成：

`[B, H, T]`

所以矩阵乘法是：

`[B, T, H] @ [B, H, T]`

结果是：

`[B, T, T]`

这就是 attention score。

那 `[T, T]` 表示什么？

它表示每个 token 和每个 token 的匹配分数。

第 i 行，表示第 i 个 token 看其他所有 token 的分数。

第 j 列，表示第 j 个 token 被当前 token 关注的程度。

如果 T 是 4，那么 attention score 是一个 4 乘 4 的矩阵。

第 0 个 token，可以看第 0、1、2、3 个 token 的分数。

第 1 个 token，也可以看第 0、1、2、3 个 token 的分数。

但是这是普通 self-attention。

对于 GPT，我们不能让它看未来。

所以要加 causal mask。

---

### 3.4 为什么除以 sqrt(H)

代码里有：

```python
/ math.sqrt(H)
```

这叫 scaled dot-product attention。

为什么要除以 `sqrt(H)`？

因为 q 和 k 的点积，如果维度 H 很大，数值容易变大。

数值太大之后，softmax 会变得非常尖锐。

比如一个位置概率接近 1，其他位置接近 0。

这会让训练不稳定。

所以除以 `sqrt(H)`，把 attention score 的尺度压回来。

你今天不需要推导方差。

你只要记住：

除以 `sqrt(head_size)` 是为了稳定 softmax 和训练。

---

### 3.5 causal mask 是什么

看代码：

```python
self.register_buffer(
    "tril",
    torch.tril(torch.ones(block_size, block_size))
)
```

这行也非常重要。

`torch.ones(block_size, block_size)` 生成一个全 1 矩阵。

`torch.tril(...)` 取下三角。

比如 4 乘 4 的下三角是：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

这就是 causal mask。

它的含义是：

第 0 个 token 只能看第 0 个 token。

第 1 个 token 可以看第 0 和第 1 个 token。

第 2 个 token 可以看第 0、第 1、第 2 个 token。

第 3 个 token 可以看第 0、第 1、第 2、第 3 个 token。

也就是只能看自己和过去，不能看未来。

---

### 3.6 register_buffer 是什么

这里还有一个容易被忽略的点：

```python
self.register_buffer("tril", ...)
```

为什么不用：

```python
self.tril = ...
```

也可以，但 `register_buffer` 更规范。

它表示：

`tril` 不是模型参数，不需要训练。

但是它是模型的一部分。

当模型 `.to(device)` 的时候，它也应该跟着移动到 GPU 或 CPU。

当模型保存 state_dict 的时候，它也可以被保存下来。

所以：

参数用 `nn.Parameter` 或者层里的权重管理。

不训练但属于模型状态的张量，用 `register_buffer`。

causal mask 就属于后者。

---

### 3.7 masked_fill 为什么填 -inf

看代码：

```python
wei_masked = wei.masked_fill(mask == 0, float("-inf"))
```

意思是：

mask 为 0 的位置，也就是未来位置，填成负无穷。

为什么是 `-inf`？

因为下一步要做 softmax。

softmax 会把数值变成概率。

如果一个位置是负无穷，那么 softmax 之后它的概率就是 0。

这就实现了禁止关注未来 token。

所以你要记住：

mask 不是把值删掉。

mask 是把未来位置的 attention score 填成 `-inf`。

softmax 之后，未来位置概率变成 0。

---

### 3.8 softmax 后是什么

代码：

```python
att = F.softmax(wei_masked, dim=-1)
```

`dim=-1` 表示沿最后一个维度做 softmax。

这里最后一个维度就是每一行里的所有 token。

所以每个 token 对所有可见 token 的关注分数，会变成一组概率。

每一行加起来等于 1。

比如第 3 个 token 可以看第 0、1、2、3 个 token。

softmax 之后可能是：

```text
0.1 0.2 0.3 0.4
```

表示第 3 个 token 对四个位置的关注权重。

而未来位置如果被 mask，就会是 0。

---

### 3.9 out = att @ v

最后：

```python
out = att @ v
```

att 是：

`[B, T, T]`

v 是：

`[B, T, H]`

结果是：

`[B, T, H]`

这是什么意思？

对于每个 token，它用 attention weight 对所有可见 token 的 value 做加权求和。

于是每个 token 的新表示，就融合了上下文信息。

所以 self-attention 的一句话总结是：

每个 token 通过 Q 和 K 计算应该关注谁，再用这个权重去加权汇总 V，得到融合上下文的新表示。

---

## 第四部分：Mini GPT

现在进入第四个文件：

`04_mini_gpt.py`

这个文件把前面的东西组装成一个最小 GPT。

结构是：

token embedding。

position embedding。

多个 Transformer block。

final LayerNorm。

LM Head。

---

### 4.1 为什么需要 position embedding

我们前面说过：

token embedding 把 token id 变成向量。

但是有一个问题：

self-attention 本身不天然知道顺序。

对于 self-attention 来说，如果没有位置信息，它只看到一堆 token 向量。

它不知道谁在前，谁在后。

但是语言顺序非常重要。

比如：

`car hits wall`

和：

`wall hits car`

token 一样，但顺序不同，意思完全不同。

所以我们需要 position embedding。

代码是：

```python
self.position_embedding_table = nn.Embedding(block_size, n_embd)
```

它的意思是：

每个位置都有一个可学习的位置向量。

位置 0 有一个向量。

位置 1 有一个向量。

位置 2 有一个向量。

一直到 `block_size - 1`。

在 forward 里：

```python
pos = torch.arange(T, device=device)
pos_emb = self.position_embedding_table(pos)
```

`torch.arange(T)` 会生成：

`[0, 1, 2, ..., T-1]`

然后查 position embedding 表，得到：

`[T, C]`

再和 token embedding 相加：

```python
x = tok_emb + pos_emb
```

token embedding 是：

`[B, T, C]`

position embedding 是：

`[T, C]`

相加时 PyTorch 会自动 broadcast，把 position embedding 扩展到 batch 维度。

结果还是：

`[B, T, C]`

你要背：

token embedding 表示“这个 token 是什么”。

position embedding 表示“这个 token 在哪里”。

---

### 4.2 Head 和 MultiHeadAttention

我们先看单个 Head。

一个 Head 就是一个单头 causal self-attention。

输入：

`[B, T, C]`

输出：

`[B, T, head_size]`

如果：

`n_embd = 64`

`n_head = 4`

那么：

`head_size = 64 // 4 = 16`

每个 head 输出 16 维。

4 个 head 拼起来，就是 64 维。

这就是 Multi-Head Attention。

为什么需要多个 head？

你可以这样理解：

一个 head 只能学一种关注模式。

多个 head 可以并行学习不同关注模式。

比如一个 head 关注前面的动作词。

一个 head 关注交通灯。

一个 head 关注车道。

一个 head 关注障碍物。

这只是直观类比，不是严格解释。

但初学时这样理解足够。

代码：

```python
out = torch.cat([head(x) for head in self.heads], dim=-1)
```

每个 head 输出 `[B, T, head_size]`

拼接之后输出 `[B, T, n_embd]`

然后：

```python
out = self.proj(out)
```

再过一个线性层，让不同 head 的信息混合。

---

### 4.3 FeedForward 是干什么的

看：

```python
self.net = nn.Sequential(
    nn.Linear(n_embd, 4 * n_embd),
    nn.ReLU(),
    nn.Linear(4 * n_embd, n_embd),
    nn.Dropout(dropout),
)
```

FeedForward 是前馈网络。

它对每个 token 位置独立作用。

也就是说，它不负责 token 之间通信。

token 之间通信是 attention 负责的。

FeedForward 的作用是对每个 token 的表示做非线性变换，提高模型表达能力。

为什么中间维度是 `4 * n_embd`？

这是 Transformer 里常见的设计。

先升维，再激活，再降维。

可以理解为让每个 token 有更强的局部变换能力。

今天你不用纠结为什么一定是 4 倍。

只要知道这是常见经验配置。

---

### 4.4 Residual Connection 是什么

在 Block 里：

```python
x = x + self.sa(self.ln1(x))
x = x + self.ffwd(self.ln2(x))
```

这里的：

```python
x = x + ...
```

就是 residual connection，残差连接。

为什么要残差连接？

因为深层网络训练时，信息和梯度都容易衰减。

残差连接相当于给信息开了一条高速公路。

原始的 x 可以直接传到后面。

attention 或 feedforward 学到的是对 x 的补充和修正。

你可以这样理解：

没有残差时，模型每层都必须重新变换 x。

有残差时，模型只需要学习“在原始 x 上加点什么”。

这通常更容易训练。

你要背：

residual connection 的作用是保留原始信息，并改善深层网络的梯度传播。

---

### 4.5 LayerNorm 是什么

Block 里还有：

```python
self.ln1 = nn.LayerNorm(n_embd)
self.ln2 = nn.LayerNorm(n_embd)
```

LayerNorm 的作用是稳定训练。

它会对每个 token 的 embedding 维度做归一化。

这里我们使用的是 Pre-LN 结构：

```python
x = x + attention(layernorm(x))
x = x + feedforward(layernorm(x))
```

为什么叫 Pre-LN？

因为 LayerNorm 在 attention 和 feedforward 之前。

还有一种是 Post-LN，就是先过子层，再做 LayerNorm。

现代 GPT 类模型常用 Pre-LN，因为训练更稳定。

今天你不需要推导 LayerNorm 公式。

你只要知道：

LayerNorm 是为了让每层输入的数值分布更稳定，从而让训练更稳定。

---

### 4.6 Dropout 是什么

代码里有：

```python
self.dropout = nn.Dropout(dropout)
```

Dropout 是一种正则化方法。

训练时，它会随机把一部分激活置零。

这样模型不能过度依赖某几个神经元。

不过今天我们的数据很小，dropout 不是重点。

你只要知道：

Dropout 训练时生效，eval 时关闭。

所以我们在估计 loss 的时候会：

```python
model.eval()
```

估计完再：

```python
model.train()
```

这两个状态很重要。

`model.train()` 表示进入训练模式，Dropout 会生效。

`model.eval()` 表示进入推理模式，Dropout 不再随机丢弃。

---

### 4.7 MiniGPT forward 全流程

现在看 MiniGPT 的 forward：

```python
B, T = idx.shape
```

idx 是 input_ids。

shape 是：

`[B, T]`

然后：

```python
tok_emb = self.token_embedding_table(idx)
```

输出：

`[B, T, C]`

然后：

```python
pos = torch.arange(T, device=device)
pos_emb = self.position_embedding_table(pos)
```

输出：

`[T, C]`

然后：

```python
x = tok_emb + pos_emb
```

输出：

`[B, T, C]`

然后：

```python
x = self.blocks(x)
```

经过多个 Transformer Block。

shape 仍然是：

`[B, T, C]`

然后：

```python
x = self.ln_f(x)
```

final LayerNorm。

shape 不变。

最后：

```python
logits = self.lm_head(x)
```

lm_head 是一个线性层。

它把 C 维 hidden state 映射到 vocab_size 维。

所以 logits 是：

`[B, T, vocab_size]`

也就是每个位置都预测下一个 token 的分数。

---

### 4.8 为什么 generate 要截断到 block_size

看生成代码：

```python
idx_cond = idx[:, -block_size:]
```

为什么要截断？

因为 position embedding 只有 `block_size` 长。

模型训练时最多只见过 `block_size` 长度的序列。

如果生成时输入越来越长，超过 block_size，position embedding 就没有对应位置了。

所以生成时只保留最后 `block_size` 个 token 作为上下文。

这就是：

```python
idx[:, -block_size:]
```

意思是取最后 block_size 个 token。

你要背：

GPT 生成时可以不断变长，但每一步实际喂给模型的上下文长度不能超过 block_size。

---

## 第五部分：训练 Mini GPT 时你应该观察什么

运行 `04_mini_gpt.py` 后，你要看几件事。

第一，看 device。

如果有 GPU，会显示 cuda。

如果没有 GPU，会显示 cpu。

第二，看 vocab_size。

这说明字符表大小。

第三，看参数量。

比如：

`number of parameters: 0.1 M`

这说明模型很小。

第四，看 shape。

你要确认：

`xb.shape = [B, T]`

`yb.shape = [B, T]`

`logits.shape = [B, T, vocab_size]`

第五，看 loss 是否下降。

一开始 loss 比较高。

训练一段时间后，train loss 应该下降。

val loss 也可能下降，但因为数据很小，不要太纠结泛化。

今天的验收标准不是模型多聪明。

今天的验收标准是：

loss 能下降。

generate 能输出类似训练文本风格的字符。

---

## 第六部分：你必须能回答的五个问题

现在我们来做最后总结。

第一个问题：

input_ids 是什么？

你要回答：

input_ids 是 tokenizer 把文本转换成的整数序列。

模型不能直接处理字符串。

模型真正看到的是 input_ids。

在字符级 tokenizer 里，每个字符对应一个整数 id。

---

第二个问题：

logits 是什么？

你要回答：

logits 是模型输出的未归一化分数。

在语言模型中，每个位置都会输出一个长度为 vocab_size 的 logits 向量。

这个向量表示当前位置对下一个 token 的预测打分。

logits 不是概率。

经过 softmax 之后才是概率。

---

第三个问题：

causal mask 为什么必要？

你要回答：

GPT 是自回归语言模型。

训练目标是根据过去 token 预测下一个 token。

如果没有 causal mask，当前位置就能看到未来 token。

这等于训练时偷看答案。

causal mask 用下三角矩阵保证第 i 个位置只能看第 0 到第 i 个位置。

---

第四个问题：

self-attention 的 `[B, T, T]` 是什么？

你要回答：

它是 attention score 或 attention weight 的 shape。

对每个 batch，每个 token 都会和序列中的每个 token 计算相关性。

所以是 `[B, T, T]`。

第 i 行表示第 i 个 token 对其他 token 的关注权重。

---

第五个问题：

next-token prediction 如何训练？

你要回答：

给定一段 token 序列，输入是前 T 个 token，目标是右移一位后的 T 个 token。

模型输出 logits，shape 是 `[B, T, vocab_size]`。

targets 的 shape 是 `[B, T]`。

训练时把 logits reshape 成 `[B*T, vocab_size]`，把 targets reshape 成 `[B*T]`。

然后用 cross entropy 计算预测分布和真实下一个 token id 的差距。

---

## 第七部分：最终总背诵版

最后，我们把今天全部内容串起来。

今天我们手写了一个字符级 GPT。

第一步，文本不能直接进模型，所以要先 tokenizer。

字符级 tokenizer 把每个字符映射成一个整数 id。

这些整数 id 组成 input_ids。

第二步，input_ids 进入 token embedding。

embedding 是一张可学习的表。

它把离散 token id 转换成连续向量。

输入是 `[B, T]`。

输出是 `[B, T, C]`。

第三步，加 position embedding。

因为 self-attention 本身不知道 token 顺序。

token embedding 表示 token 是什么。

position embedding 表示 token 在哪里。

第四步，进入 Transformer Block。

每个 Block 里面有 causal self-attention、feed forward、LayerNorm 和 residual connection。

第五步，self-attention 里面，每个 token 会生成 Q、K、V。

Q 表示我想找什么信息。

K 表示我有什么信息可以被匹配。

V 表示如果我被关注，我实际贡献什么内容。

第六步，用 Q 和 K 做点积，得到 attention score。

q 的 shape 是 `[B, T, H]`。

k 转置后是 `[B, H, T]`。

所以 attention score 是 `[B, T, T]`。

第七步，GPT 需要 causal mask。

causal mask 是下三角矩阵。

它保证当前位置只能看自己和过去，不能看未来。

未来位置被填成 `-inf`。

softmax 后，未来位置概率变成 0。

第八步，attention weight 乘以 V。

这样每个 token 就能汇总自己可见上下文的信息。

第九步，多个 attention head 拼接起来，就是 multi-head attention。

不同 head 可以学习不同的关注模式。

第十步，feed forward 对每个 token 独立做非线性变换。

attention 负责 token 之间通信。

feed forward 负责每个 token 自己的特征变换。

第十一步，residual connection 保留原始信息，并帮助梯度传播。

LayerNorm 稳定训练。

第十二步，最后 LM Head 把 hidden state 映射成 logits。

hidden state 是 `[B, T, C]`。

logits 是 `[B, T, vocab_size]`。

第十三步，训练时用 cross entropy。

每个位置都要预测下一个 token。

输入是当前 token 序列。

target 是右移一位后的 token 序列。

第十四步，生成时只能逐 token 生成。

每次取最后一个位置的 logits。

经过 softmax 得到概率。

采样一个 token。

拼回输入。

再继续预测下一个 token。

所以今天你要真正记住一句话：

GPT 不是魔法。

GPT 的最小闭环就是：

tokenizer 把文本变成 input_ids。

embedding 把 input_ids 变成向量。

causal self-attention 让 token 只能看过去上下文。

Transformer block 不断更新 token 表示。

LM Head 输出 logits。

cross entropy 训练模型预测下一个 token。

generate 时逐 token 采样生成文本。

这就是今天 Day 2 的核心。
