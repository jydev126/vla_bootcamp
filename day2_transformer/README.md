
# Day 2 Transformer / Attention / Causal LM

## 1. input_ids 是什么？

input_ids 是 tokenizer 把文本转换成的整数序列。

模型不能直接处理字符串，只能处理数字张量。

例如字符级 tokenizer 中：

```text
"the" -> [id_t, id_h, id_e]
````

这里的 `[id_t, id_h, id_e]` 就是 input_ids。

input_ids 本身没有语义。
它只是查 embedding table 的索引。

---

## 2. tokenizer 是什么？

tokenizer 不是模型。

tokenizer 的作用是把离散文本符号转换成整数 id。

在字符级 tokenizer 中，每个字符是一个 token。

在真实大模型中，常用的是 BPE、SentencePiece 等子词 tokenizer。
但本质仍然是：

```text
文本 -> token id
token id -> 文本
```

---

## 3. embedding 是什么？

embedding 是一张可学习的表。

输入 token id，输出一个向量。

例如：

```text
input_ids: [B, T]
embedding output: [B, T, C]
```

其中：

```text
B = batch size
T = sequence length
C = embedding dimension
```

embedding 的作用是把离散 token id 转换成连续向量。

---

## 4. logits 是什么？

logits 是模型输出的未归一化分数。

在语言模型中，每个位置都要预测下一个 token。

所以 logits 的 shape 是：

```text
[B, T, vocab_size]
```

含义是：

```text
每个 batch，
每个 token 位置，
都输出一个对整个词表的预测分数。
```

logits 不是概率。

logits 经过 softmax 后才是概率分布。

---

## 5. cross entropy 是什么？

cross entropy 用来衡量预测分布和真实类别之间的差距。

在 next-token prediction 中：

```text
模型输出 logits
真实答案是下一个 token id
```

cross entropy 会让正确 token 的概率越来越高。

训练时通常需要 reshape：

```text
logits:  [B, T, vocab_size] -> [B*T, vocab_size]
targets: [B, T]             -> [B*T]
```

然后计算：

```python
loss = F.cross_entropy(logits_flat, targets_flat)
```

---

## 6. next-token prediction 如何训练？

给定一段 token 序列：

```text
x = [t0, t1, t2, t3]
```

训练目标是：

```text
y = [t1, t2, t3, t4]
```

也就是：

```text
看到 t0，预测 t1
看到 t1，预测 t2
看到 t2，预测 t3
看到 t3，预测 t4
```

这就是 next-token prediction。

GPT 的训练目标就是这个。

---

## 7. Q / K / V 是什么？

在 self-attention 中，每个 token 的向量 x 会被映射成三个向量：

```text
Q = Query
K = Key
V = Value
```

可以这样理解：

```text
Q：我想找什么信息？
K：我有什么信息可以被别人匹配？
V：如果别人关注我，我实际提供什么内容？
```

计算过程是：

```text
score = Q @ K.T
weight = softmax(score)
out = weight @ V
```

---

## 8. self-attention 的 [B, T, T] 是什么？

输入 x 的 shape 是：

```text
[B, T, C]
```

经过线性层得到：

```text
q: [B, T, H]
k: [B, T, H]
v: [B, T, H]
```

attention score 是：

```text
q @ k.transpose(-2, -1)
```

所以 shape 是：

```text
[B, T, H] @ [B, H, T] = [B, T, T]
```

这个 `[B, T, T]` 表示：

```text
每个 batch 中，
每个 token 对每个 token 的关注分数。
```

第 i 行表示第 i 个 token 看其他 token 的权重。

---

## 9. causal mask 为什么必要？

GPT 是自回归语言模型。

它只能根据过去 token 预测未来 token。

如果没有 causal mask，当前位置就可以看到未来 token。

这相当于训练时偷看答案。

causal mask 是一个下三角矩阵：

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

第 i 行只能看到第 0 到第 i 个位置。

未来位置会被填成 `-inf`。

经过 softmax 后，未来位置的概率变成 0。

---

## 10. 为什么 decode 只能逐 token？

GPT 的生成是自回归的。

每一步只能生成一个新 token。

流程是：

```text
已有 token -> 预测下一个 token
拼接新 token -> 再预测下一个 token
继续循环
```

因为第 t+1 个 token 的生成依赖第 t 个 token。

所以不能一次性并行生成完整句子。

训练可以并行，因为训练时完整答案已经存在，并且 causal mask 防止模型看未来。

生成不能完全并行，因为未来 token 还没有生成出来。

---

## 11. Transformer Block 是什么？

一个最小 Transformer Block 包含：

```text
LayerNorm
Causal Self-Attention
Residual Connection
LayerNorm
Feed Forward Network
Residual Connection
```

常见 Pre-LN 写法是：

```text
x = x + attention(layernorm(x))
x = x + feedforward(layernorm(x))
```

attention 负责 token 之间的信息交流。

feed forward 负责每个 token 自己的非线性变换。

residual connection 负责保留原始信息并改善梯度传播。

LayerNorm 负责稳定训练。

---

## 12. 今天最终链路

完整链路是：

```text
文本
  ↓
tokenizer
  ↓
input_ids: [B, T]
  ↓
token embedding: [B, T, C]
  ↓
position embedding: [B, T, C]
  ↓
Transformer blocks: [B, T, C]
  ↓
LM Head
  ↓
logits: [B, T, vocab_size]
  ↓
cross entropy
  ↓
next-token prediction
```

生成链路是：

```text
已有 token ids
  ↓
模型输出 logits
  ↓
取最后一个位置 logits
  ↓
softmax
  ↓
采样下一个 token
  ↓
拼回输入
  ↓
循环
```



---

# 最后你今天必须背下来的总版


今天的任务是手写一个字符级 GPT。

```text
第一步，tokenizer 把文本转换成 input_ids。
input_ids 是整数序列，不是语义向量。

第二步，embedding 把 input_ids 转换成连续向量。
输入是 [B,T]，输出是 [B,T,C]。

第三步，加 position embedding。
因为 self-attention 本身不知道 token 顺序。

第四步，进入 Transformer Block。
Block 里有 causal self-attention、feed forward、LayerNorm、residual connection。

第五步，self-attention 里，每个 token 生成 Q、K、V。
Q 表示我想找什么。
K 表示我有什么信息可被匹配。
V 表示我真正提供什么内容。

第六步，attention score = Q @ K.T。
q 是 [B,T,H]，k 转置后是 [B,H,T]。
所以 score 是 [B,T,T]。

[B,T,T] 表示每个 token 对每个 token 的关注分数。

第七步，GPT 必须使用 causal mask。
causal mask 是下三角矩阵。
它保证当前位置只能看自己和过去，不能看未来。
未来位置填 -inf，softmax 后概率变成 0。

第八步，attention weight 乘以 V，得到每个 token 汇总上下文后的表示。

第九步，经过 LM Head，把 [B,T,C] 映射成 [B,T,vocab_size]。
这个输出叫 logits。

第十步，logits 是未归一化分数，不是概率。
经过 softmax 才是概率。

第十一步，训练目标是 next-token prediction。
输入是当前 token 序列，target 是右移一位后的 token 序列。

第十二步，cross entropy 衡量预测分布和真实下一个 token id 的差距。

第十三步，生成时只能逐 token 生成。
每次取最后一个位置的 logits，采样一个 token，拼回输入，再继续预测。
```

你今天真正要吃透的不是“Transformer 很复杂”，而是这一句话：

```text
GPT = tokenizer + embedding + causal self-attention + next-token prediction。
```
