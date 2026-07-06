# 07 完整笔记｜Let's build GPT：Transformer、Attention 和 Causal LM

## 0. 这节课到底要学会什么

这节是整个 Zero to Hero 系列的主干：从零实现一个 GPT。

核心链路：

```text
文本
-> token ids
-> token embedding + position embedding
-> 多层 Transformer block
-> logits
-> cross entropy(next token)
-> 自回归生成
```

你学 VLA 必须吃透这节，因为现代 VLA 往往不是从零写控制网络，而是站在 LLM/VLM/Transformer backbone 上，把视觉、语言、状态、动作都变成 token/hidden，然后接 planning/action。

## 1. GPT 是什么

GPT = Generative Pretrained Transformer。

本节实现的是简化 GPT，不是 ChatGPT。

它的基础行为：

```text
给定一段 token 序列，预测下一个 token 的概率分布。
```

生成时：

```text
不断预测下一个 token，再把预测结果接回上下文。
```

这就是自回归生成。

## 2. 数据：Tiny Shakespeare

课程用 Shakespeare 文本训练字符级 GPT。

文本先被转成字符表：

```text
chars = sorted(list(set(text)))
vocab_size = len(chars)
```

建立：

```text
stoi: char -> int
itos: int -> char
```

encode/decode：

```text
encode("hello") -> [id_h, id_e, id_l, id_l, id_o]
decode(ids) -> "hello"
```

这里是字符级 tokenizer。真实 GPT 用 BPE tokenizer，但建模目标一样。

## 3. train/val split

把长文本 token 序列切成：

```text
train_data
val_data
```

训练用 train，调试泛化用 val。

这和 VLA 数据集也一样：不能只看训练路线，要看 held-out 场景、长尾场景、闭环仿真。

## 4. block_size

`block_size` 是模型一次能看的最大上下文长度。

如果：

```text
block_size = 8
```

模型最多根据前 8 个 token 预测后续 token。

GPT 的上下文窗口本质就是更大的 block_size。

VLA 里上下文可能包括：

```text
图像 tokens
文本 instruction tokens
历史轨迹 tokens
动作历史 tokens
latent/action query tokens
```

上下文越长，计算越贵。

## 5. get_batch

训练样本来自长 token 序列的随机片段。

如果取一段：

```text
x = data[i : i + block_size]
y = data[i + 1 : i + block_size + 1]
```

那么：

```text
x[t] 的目标是 y[t]，也就是原序列的下一个 token
```

batch 后：

```text
xb.shape = [B, T]
yb.shape = [B, T]
```

`B` 是 batch size，`T` 是 block_size。

## 6. 为什么 `x` 和 `y` 要 shift

例子：

```text
text: h e l l o
x:    h e l l
y:    e l l o
```

模型在位置 0 看到 `h`，预测 `e`。
模型在位置 1 看到 `h e`，预测 `l`。
模型在位置 2 看到 `h e l`，预测 `l`。

这叫 next-token prediction。

Action-token VLA 里也一样：

```text
action_in  = BOS, a0, a1, a2
action_out = a0,  a1, a2, a3
```

## 7. BigramLanguageModel 作为 baseline

最简单模型：

```text
logits = token_embedding_table(idx)
```

如果：

```text
idx.shape = [B, T]
embedding_table.shape = [V, V]
```

输出：

```text
logits.shape = [B, T, V]
```

这个模型本质还是 bigram：每个位置只根据当前 token 预测下一个 token。

它用来建立训练/生成框架。

## 8. logits reshape 和 cross entropy

PyTorch `F.cross_entropy` 期望：

```text
input:  [N, C]
target: [N]
```

但语言模型 logits 是：

```text
[B, T, V]
```

所以 reshape：

```text
logits = logits.view(B*T, V)
targets = targets.view(B*T)
loss = F.cross_entropy(logits, targets)
```

这意味着每个 batch、每个时间位置都是一个分类样本。

## 9. 生成 generate

生成时：

```text
idx: [B, current_T]
logits = model(idx)
last_logits = logits[:, -1, :]
probs = softmax(last_logits)
next_id = sample(probs)
idx = cat(idx, next_id)
```

为什么只取最后一个位置？

因为要生成下一个 token，只需要当前序列最后位置对 next token 的预测。

## 10. self-attention 的动机

Bigram 只看当前 token。我们希望每个 token 能根据前面的 token 更新自己的表示。

Self-attention 做的事：

```text
每个 token 根据 query 去匹配其他 token 的 key，再读取它们的 value。
```

这让模型可以动态决定关注上下文中的哪些位置。

## 11. Q/K/V

输入 hidden：

```text
x.shape = [B, T, C]
```

线性投影：

```text
q = x @ Wq
k = x @ Wk
v = x @ Wv
```

shape：

```text
q, k, v: [B, T, H]
```

直觉：

```text
Query: 我在找什么
Key: 我有什么索引/标签
Value: 我真正提供什么内容
```

## 12. attention score

计算：

```text
wei = q @ k.transpose(-2, -1)
```

shape：

```text
[B, T, H] @ [B, H, T] -> [B, T, T]
```

`wei[b, i, j]` 表示第 i 个 token 对第 j 个 token 的关注分数。

## 13. 为什么除以 sqrt(head_size)

如果 q/k 维度很大，dot product 方差会变大，softmax 容易变得极端。

所以缩放：

```text
wei = wei / sqrt(H)
```

这让 softmax 更稳定，梯度更健康。

这和第 4 节初始化/激活尺度是同一类问题。

## 14. causal mask

GPT 不能看未来 token。

所以对 attention matrix 做下三角 mask：

```text
未来位置 -> -inf
softmax 后 -> 0
```

如果没有 causal mask，训练时模型会直接偷看答案，loss 很低但生成无效。

VLA action token 也必须 mask 未来动作。

## 15. softmax 和加权求和

mask 后：

```text
att = softmax(wei, dim=-1)
out = att @ v
```

shape：

```text
att: [B, T, T]
v:   [B, T, H]
out: [B, T, H]
```

每个位置输出的是它对可见上下文 value 的加权汇总。

## 16. Multi-head attention

一个 head 只能在一个子空间里看关系。多个 head 并行：

```text
head1(x), head2(x), ..., headN(x)
concat -> projection
```

不同 head 可以学不同关系：

```text
局部字符模式
长距离依赖
括号匹配
语法结构
命令和对象关系
```

VLA 中不同 head 可能关注车道线、前车、导航命令、障碍物、历史轨迹等。

## 17. FeedForward

Attention 负责 token 间通信。

FeedForward 负责每个 token 内部的非线性变换：

```text
Linear(C, 4C)
ReLU/GELU
Linear(4C, C)
```

它独立作用在每个 token position 上。

## 18. Residual connection

Transformer block 使用残差：

```text
x = x + attention(norm(x))
x = x + feedforward(norm(x))
```

残差让梯度有更直接路径，深层网络更容易训练。

没有 residual，深层 Transformer 很难优化。

## 19. LayerNorm

LayerNorm 对每个 token 的 hidden 维做归一化。

和 BatchNorm 不同：

```text
BatchNorm: 依赖 batch 统计
LayerNorm: 对单个样本/单个 token 的 channel 维统计
```

Transformer 常用 LayerNorm，因为序列长度、batch size、生成模式都更适合它。

## 20. Dropout

Dropout 训练时随机丢弃部分激活，防止过拟合。

推理时关闭。

所以必须区分：

```text
model.train()
model.eval()
```

## 21. Transformer block 总结构

典型 block：

```text
x -> LayerNorm -> MultiHeadAttention -> residual add
  -> LayerNorm -> FeedForward -> residual add
```

堆叠多个 block：

```text
x -> block1 -> block2 -> ... -> blockN -> final LayerNorm -> LM head
```

## 22. position embedding

Self-attention 本身不天然知道顺序。

如果没有 position embedding，模型只看到一组 token，不知道谁在前谁在后。

所以输入是：

```text
tok_emb = token_embedding(idx)
pos_emb = position_embedding(arange(T))
x = tok_emb + pos_emb
```

VLA 中也有类似需求：

```text
图像 patch 位置
时间步位置
历史轨迹顺序
future waypoint index
action token index
```

## 23. 最终 LM head

Transformer 输出：

```text
x.shape = [B, T, C]
```

LM head：

```text
logits = x @ W_vocab
```

shape：

```text
[B, T, V]
```

每个位置都预测下一个 token。

## 24. estimate_loss

训练时不要每步都只看一个 batch 的 loss，因为噪声大。

可以定期在 train/val 上评估多个 batch 平均 loss：

```text
model.eval()
with torch.no_grad():
    average loss over eval_iters
model.train()
```

这也是 VLA 实验的基本习惯：定期验证，不只看训练 batch。

## 25. nanoGPT 连接

课程最后会指向 nanoGPT。你手写的 mini GPT 和 nanoGPT 结构一致：

```text
Config
CausalSelfAttention
MLP
Block
GPT
training loop
generate
```

区别是 nanoGPT 更工程化、更高效、更适合真实训练。

## 26. 从 GPT 到 ChatGPT 的区别

这节训练的是 base language model。

ChatGPT 还需要：

```text
pretraining
supervised fine-tuning
preference / reward modeling
RLHF 或其他 alignment
```

所以不要把“会 next-token prediction”直接等同于“会当助手”。

## 27. 对 VLA 全栈的意义

VLA 可以用 GPT/Transformer 思想做几件事：

```text
1. 文本 instruction 编码
2. 图像 token 和文本 token 融合
3. 状态/history token 融合
4. latent/action query 从上下文读取信息
5. action token 自回归生成
6. hidden state 接 continuous trajectory head
```

你要会画：

```text
input tokens -> transformer hidden -> selected hidden -> task head
```

## 28. continuous head 和 action-token head

两种 VLA 输出路线：

```text
continuous:
selected hidden -> MLP -> waypoints/control

discrete:
previous action tokens -> Transformer -> action logits -> sample/decode
```

两者都依赖 Transformer hidden。区别是输出空间和 loss 不同。

## 29. prefill/decode 的前置理解

这节 generate 是最简单版本，每次把整个上下文重新 forward。

真实 LLM 会缓存 KV：

```text
prefill: 一次性处理 prompt/context
decode: 每次只处理新 token，复用 KV cache
```

VLA 部署关心延迟时，这非常重要。

## 30. 复习自测

1. `x` 和 `y` 为什么要错位？
2. `logits` 为什么要 reshape 后算 cross entropy？
3. self-attention 的 Q/K/V 分别是什么？
4. attention score 为什么是 `[B, T, T]`？
5. 为什么要除以 `sqrt(head_dim)`？
6. causal mask 防止什么问题？
7. Multi-head attention 比 single head 强在哪里？
8. FeedForward 在 Transformer 里做什么？
9. Residual 和 LayerNorm 为什么重要？
10. position embedding 为什么必要？
11. 生成时为什么只取最后一个位置的 logits？
12. VLA 中哪些 token 需要位置/时间信息？
13. action-token VLA 为什么需要 causal mask？
14. continuous trajectory head 和 LM head 的区别是什么？
