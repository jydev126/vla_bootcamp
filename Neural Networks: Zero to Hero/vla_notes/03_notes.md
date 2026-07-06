# 03 完整笔记｜makemore MLP：embedding、上下文窗口和可训练表示空间

## 0. 这节课到底要学会什么

上一节 bigram 只看一个字符。这节课升级到 MLP 语言模型：给模型多个历史字符，让它预测下一个字符。

核心链路：

```text
context token ids
-> embedding table 查向量
-> 多个 token embedding 拼接
-> hidden layer + tanh
-> logits
-> cross entropy
-> 反向传播训练 embedding 和 MLP 参数
```

这节对 VLA 很关键，因为它第一次真正把“离散 token”变成“可训练向量表示”。后面所有 VLA 里的 text token、visual token、state token、map token、action token、latent slot，本质都要进入某种 embedding/hidden space。

## 1. bigram 为什么不够

bigram 建模：

```text
P(x_t | x_{t-1})
```

也就是只看前一个字符。

问题：很多规律依赖更长上下文。

例如名字里：

```text
emma
emily
ethan
```

当你预测下一个字符时，只看最后一个字符可能不够。`m` 后面接什么，可能和前面是 `e`、`a`、`o` 有关。

如果用计数表扩展到三个字符上下文：

```text
P(x_t | x_{t-3}, x_{t-2}, x_{t-1})
```

表大小会变成：

```text
27 * 27 * 27 * 27
```

上下文再长就指数爆炸，而且很多组合在训练集中根本没出现。计数表无法泛化。

神经网络的方案：把离散上下文映射到连续向量空间，用参数共享来泛化。

## 2. Bengio 2003 神经语言模型的思想

这节参考的是早期神经概率语言模型：

```text
用前 n 个词/字符预测下一个词/字符
每个 token 先查 embedding
把 embedding 输入神经网络
输出下一个 token 的概率分布
```

关键突破：不再为每个上下文组合单独计数，而是让 token 有共享的向量表示。

如果两个 token 在功能上相似，它们的 embedding 可以学得接近，于是模型能在未见过的组合上泛化。

## 3. 构造 dataset

设：

```text
block_size = 3
```

表示用前 3 个字符预测下一个字符。

对名字 `emma`，先加边界：

```text
emma.
```

上下文初始为：

```text
[0, 0, 0]
```

其中 0 对应 `.`。

逐步构造样本：

```text
context: ... -> target: e
context: ..e -> target: m
context: .em -> target: m
context: emm -> target: a
context: mma -> target: .
```

每生成一个样本，就把 target 滑入 context：

```text
context = context[1:] + [target]
```

这就是固定窗口语言模型。

## 4. `X` 和 `Y` 的 shape

训练集最终是：

```text
X.shape = [num_examples, block_size]
Y.shape = [num_examples]
```

如果 `block_size = 3`：

```text
X[i] = 三个历史 token id
Y[i] = 下一个 token id
```

例子：

```text
X[i] = [0, 5, 13]
Y[i] = 13
```

这里 `[0, 5, 13]` 不是连续值特征，而是 token id。必须先通过 embedding table。

## 5. embedding table `C`

定义：

```text
C.shape = [vocab_size, embedding_dim]
```

如果 vocab size 是 27，embedding dim 是 2：

```text
C.shape = [27, 2]
```

每一行是一个字符的可训练向量。

索引：

```text
emb = C[X]
```

如果：

```text
X.shape = [B, 3]
```

那么：

```text
emb.shape = [B, 3, 2]
```

这里 `B` 是 batch size，`3` 是上下文长度，`2` 是每个 token 的向量维度。

## 6. 为什么 embedding lookup 可以训练

`C[X]` 看起来只是查表，但它在 PyTorch 里是可微的。

forward：取出被用到的行。

backward：loss 的梯度会回到这些被取出的行。

如果某个字符在 batch 中出现很多次，它那一行 embedding 会收到多次梯度累加。

这和 VLA 里的 action token embedding、special token embedding、latent token embedding 都一样。

## 7. flatten 上下文

MLP 需要一个向量输入，所以把多个 token embedding 拼起来：

```text
emb.shape = [B, block_size, embedding_dim]
emb_flat.shape = [B, block_size * embedding_dim]
```

例如：

```text
[B, 3, 2] -> [B, 6]
```

这一步的意义：把 3 个字符的向量表示合成一个上下文表示。

## 8. 第一层 hidden layer

参数：

```text
W1.shape = [block_size * embedding_dim, hidden_dim]
b1.shape = [hidden_dim]
```

计算：

```text
h_pre = emb_flat @ W1 + b1
h = tanh(h_pre)
```

如果：

```text
emb_flat.shape = [B, 6]
W1.shape = [6, 100]
```

则：

```text
h.shape = [B, 100]
```

这 100 维 hidden 是模型对当前上下文的内部表示。

## 9. 输出层 logits

参数：

```text
W2.shape = [hidden_dim, vocab_size]
b2.shape = [vocab_size]
```

计算：

```text
logits = h @ W2 + b2
```

shape：

```text
[B, hidden_dim] @ [hidden_dim, vocab_size] -> [B, vocab_size]
```

每一行表示当前样本对所有下一个字符的打分。

## 10. cross entropy

训练目标：让真实下一个字符的 logit 对应概率变高。

手写可以做：

```text
counts = logits.exp()
probs = counts / counts.sum(dim=1, keepdim=True)
loss = -probs[range(B), Y].log().mean()
```

PyTorch 更推荐：

```python
loss = F.cross_entropy(logits, Y)
```

原因：

```text
数值更稳定
内部处理 log-softmax
不会轻易 exp 溢出
```

## 11. 参数集合

这个 MLP 的参数包括：

```text
C
W1
b1
W2
b2
```

训练时这些都要：

```text
requires_grad = True
```

每步训练：

```text
forward
loss
zero grad
backward
update
```

注意 embedding table `C` 也是参数。模型不只是学 MLP 权重，也在学字符表示空间。

## 12. mini-batch 训练

全量训练每一步用所有样本，太慢。

mini-batch 做法：

```text
随机采样 B 个样本
只用这 B 个样本估计梯度
更新参数
```

代码形式：

```python
ix = torch.randint(0, X.shape[0], (batch_size,))
Xb = X[ix]
Yb = Y[ix]
```

优点：

```text
速度快
可以训练更多 step
梯度噪声有时帮助跳出坏区域
```

缺点：

```text
gradient 有噪声
loss 曲线会抖
```

## 13. 学习率怎么找

课程里会尝试不同 learning rate。

学习率太小：

```text
loss 降得很慢
```

学习率太大：

```text
loss 发散或震荡
```

一种做法是学习率扫描：

```text
从很小 lr 到很大 lr
观察 loss 什么时候开始快速下降
什么时候开始不稳定
```

这对大模型也非常重要。VLA 里 projector、LLM adapter、action head 的学习率可能不一样。

## 14. train/dev/test split

不能只看训练集 loss。

划分：

```text
train: 更新参数
dev/val: 调超参数，观察泛化
test: 最终评估，只用一次或少量使用
```

如果：

```text
train loss 低，dev loss 高
```

说明过拟合。

如果：

```text
train loss 和 dev loss 都高
```

说明模型容量不够、训练不够、优化不好或特征不够。

## 15. 模型容量

可以调：

```text
embedding_dim
hidden_dim
block_size
训练 step
learning rate
```

容量太小：欠拟合。

容量太大：可能过拟合，也更难训练。

课程会逐步调整这些配置，让 loss 改善。

## 16. 为什么 embedding 可以可视化

如果 embedding dim 是 2，就可以把每个字符画在二维平面上。

训练后，模型可能把相似作用的字符放得更近。

例如元音可能形成某种聚集，或者某些字符由于上下文功能相似而靠近。

这说明 embedding 不只是查表编号，而是模型学到的表示空间。

## 17. 生成过程

训练好后生成名字：

```text
context = [0, 0, 0]
while True:
    emb = C[context]
    h = tanh(emb_flat @ W1 + b1)
    logits = h @ W2 + b2
    probs = softmax(logits)
    ix = sample(probs)
    if ix == 0: break
    append ix
    context = context[1:] + [ix]
```

注意生成时模型每次只预测一个 token，然后把生成结果放回上下文。

这是自回归生成的雏形。

## 18. 这节课里的 shape 总表

假设：

```text
B = batch size
T = block_size
E = embedding_dim
H = hidden_dim
V = vocab_size
```

则：

```text
Xb:        [B, T]
Yb:        [B]
C:         [V, E]
emb:       [B, T, E]
emb_flat:  [B, T*E]
W1:        [T*E, H]
b1:        [H]
h:         [B, H]
W2:        [H, V]
b2:        [V]
logits:    [B, V]
loss:      scalar
```

这张表要反复看。后面 Transformer 只是把 `[B, T, C]` 保留得更久，不急着 flatten。

## 19. 这节课的关键抽象

从 bigram 到 MLP，真正的升级不是“多了一层网络”，而是三个抽象：

```text
1. token id 可以映射成可训练向量
2. 多个 token 的上下文可以组合成 hidden representation
3. hidden representation 可以接一个 head 预测目标
```

这三个抽象会贯穿 VLA。

## 20. 和 VLM 的连接

VLM 里的图像也要变成 token-like 表示：

```text
image
-> patches
-> vision encoder
-> visual features
-> projector
-> visual tokens
```

这和字符 embedding 的精神类似：原始离散/非结构化输入不能直接给语言模型，需要变成 hidden vectors。

区别：

```text
字符 token: 通过 embedding table 查向量
图像 patch: 通过 vision encoder 算向量
```

但进入 Transformer 后，都变成 `[B, T, C]` 的 token sequence。

## 21. 和 action 建模的连接

VLA 的 action 有两种主要路线。

连续路线：

```text
multimodal hidden -> MLP/head -> continuous waypoints/control
```

离散 token 路线：

```text
continuous action -> discretize -> action token id -> action embedding -> LM predicts next action token
```

这节的 embedding table 直接支撑第二种路线。

## 22. 和 latent/action query 的连接

后面的 latent token 或 action query 可以理解成特殊的可训练向量槽位。

它们不是来自文本词表里的普通词，而是参数：

```text
latent_slots.shape = [num_latents, hidden_dim]
action_queries.shape = [num_future_steps, hidden_dim]
```

训练后，它们学会从上下文 hidden 中吸收对任务有用的信息。

这和 `C` 里的字符 embedding 很像：初始随机，靠 loss 训练出意义。

## 23. 对完整 VLA 能力栈的意义

这节课支撑的能力不只是语言建模，而是表示学习。

VLA 需要统一很多来源：

```text
camera image
BEV/map
navigation command
ego state
history trajectory
text instruction
action history
```

每个来源都要进入某种向量空间，然后融合。

你以后读 VLA 时要问：

```text
这个输入是怎么变成 embedding/feature 的？
shape 是多少？
它和其他模态是否在同一个 hidden_dim？
有没有 projector？
有没有 position/time embedding？
最终哪个 hidden 被 head 使用？
```

## 24. 常见误区

### 24.1 以为 token id 本身有大小意义

`stoi['a'] = 1`，`stoi['z'] = 26` 不代表 z 比 a 大。id 只是索引。

意义来自 embedding 行，而不是 id 数值。

### 24.2 忘记 embedding 也是参数

如果 embedding 不训练，模型只能用随机表示，性能会差很多。

### 24.3 flatten 过早丢失结构

MLP 把 `[B, T, E]` flatten 成 `[B, T*E]`，简单但不够优雅。Transformer 会保留 token 维度，让 token 间通过 attention 交互。

### 24.4 只看 train loss

VLA 中也一样，只看训练 loss 没意义。要看 validation、闭环指标、失败案例。

## 25. 复习自测

你应该能回答：

1. 为什么 bigram 上下文太短？
2. 为什么不用高阶计数表直接解决长上下文？
3. `block_size` 是什么？
4. `X` 和 `Y` 分别是什么 shape？
5. embedding table `C` 的 shape 是什么？
6. `C[X]` 后 shape 怎么变？
7. 为什么 embedding lookup 可以被训练？
8. flatten 后进入 MLP 的 shape 是什么？
9. logits 的最后一维为什么等于 vocab size？
10. 为什么 `F.cross_entropy` 比手写 softmax 更稳定？
11. train/dev/test 各自用途是什么？
12. mini-batch 为什么会让 loss 曲线抖动？
13. embedding 可视化说明了什么？
14. 这节和 VLM visual token 有什么关系？
15. 这节和 action token / latent slot 有什么关系？
