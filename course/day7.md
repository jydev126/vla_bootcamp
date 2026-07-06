# Day 7 上课台词：OpenVLA-style action token

## 开场

今天是 Day 7。

昨天我们讲了 ACT-style continuous action。

昨天的路线是：

```text
observation tokens -> hidden state -> action queries -> continuous action chunk
```

也就是模型直接输出连续数值。

比如输出：

```text
[B, 6, 2]
```

代表 6 个未来 waypoint。

今天我们讲另一条路线。

这条路线的代表是 OpenVLA-style action token。

它的核心问题是：

既然 LLM 最擅长生成 token，那能不能把连续动作也变成 token？

如果可以，那么 action prediction 就可以被改写成 language modeling。

也就是说：

```text
连续动作 -> 离散 action token ids
context tokens -> causal transformer -> action token logits
```

这和 Day2 的 mini GPT 非常接近。

Day2 是根据前面的文本 token 预测下一个文本 token。

今天是根据 observation context 和之前的 action token，预测下一个 action token。

今天两个脚本：

```bash
python day7_openvla_action/01_action_tokenizer.py
python day7_openvla_action/02_tiny_action_lm.py
```

今天你要掌握到什么程度？

第一，知道连续动作为什么需要离散化才能变成 token。

第二，知道 bins 是什么，bins 太多太少有什么问题。

第三，知道 action LM 里的 shift 和 causal mask 为什么和 Day2 一样重要。

第四，知道 action token 路线和 ACT continuous 路线的取舍。

---

## 第一部分：为什么要把 action 变成 token

我们先从动机讲。

LLM 的训练目标是 next-token prediction。

也就是：

```text
给定前面的 token，预测下一个 token。
```

如果我们希望用 LLM 的方式输出动作，就要把动作也表示成 token。

可是动作通常是连续数值。

比如一个 waypoint 坐标可能是：

```text
0.137
-0.042
0.281
```

这些不是离散词表里的 token。

所以 OpenVLA-style 的想法是：

把连续动作离散化。

把每个连续值映射到一个整数 id。

这样动作就变成 action token ids。

然后模型就可以用 cross entropy 来预测。

你可以把它理解成：

```text
把回归问题变成分类问题。
```

这句话很重要。

连续动作回归用 MSE。

离散 action token 预测用 CrossEntropy。

---

## 第二部分：01_action_tokenizer.py，连续值怎么变成 id

第一个脚本是：

```text
day7_openvla_action/01_action_tokenizer.py
```

它有两个函数：

```python
continuous_to_action_ids
action_ids_to_continuous
```

先看连续动作。

`future_to_tensor(sample)` 输出 shape 是：

```text
[12]
```

这 12 个数来自 6 个 waypoint。

代码里假设这些数大致归一化在 -1 到 1。

离散化公式是：

```python
ids = torch.round((target.clamp(-1, 1) + 1.0) * 0.5 * (bins - 1))
```

我们一步一步拆。

第一步：

```python
target.clamp(-1, 1)
```

把连续值限制在 -1 到 1。

为什么要 clamp？

因为离散化的 bins 只覆盖这个范围。

如果有值超出范围，就先截断。

第二步：

```python
+ 1.0
```

把范围从 `[-1, 1]` 变成 `[0, 2]`。

第三步：

```python
* 0.5
```

把范围变成 `[0, 1]`。

第四步：

```python
* (bins - 1)
```

把范围变成 `[0, bins - 1]`。

第五步：

```python
round(...).long()
```

把连续位置变成整数 id。

如果 bins 是 32，那么 id 范围是：

```text
0, 1, 2, ..., 31
```

所以一个 12 维连续 action，会变成 12 个 action token ids。

```text
continuous trajectory: [12]
action token ids: [12]
```

---

## 第三部分：detokenize 为什么只能近似恢复

脚本里还有反向函数：

```python
action_ids_to_continuous
```

它做：

```python
ids.float() / max(bins - 1, 1) * 2.0 - 1.0
```

这会把 id 映射回 -1 到 1 的连续值。

但注意，只能近似恢复。

因为离散化一定丢信息。

假设 bins 是 32。

-1 到 1 之间只有 32 个格子。

很多不同的连续值会落到同一个格子。

比如 0.137 和 0.151 可能都变成同一个 action id。

变回来时，它们都会得到同一个近似值。

所以 action tokenization 有代价。

它的好处是可以用 LM 框架。

它的坏处是有量化误差。

bins 太少，量化误差大。

bins 太多，分类难度大。

这就是你以后看 OpenVLA 时必须理解的 tradeoff。

---

## 第四部分：02_tiny_action_lm.py 的整体任务

第二个脚本是：

```text
day7_openvla_action/02_tiny_action_lm.py
```

它实现一个 tiny action language model。

Dataset 返回：

```text
context_ids
mask
action_ids
```

context_ids 是状态 token。

比如：

```text
BOS speed_4 dist_2 rel_1 cmd_brake ... EOS
```

action_ids 是未来 trajectory 离散化后的 12 个 action token。

模型任务是：

```text
给定 context tokens 和之前的 action tokens，预测下一个 action token。
```

这和 Day2 的 causal LM 完全对应。

Day2：

```text
文本前缀 -> 下一个文本 token
```

Day7：

```text
状态前缀 + 动作前缀 -> 下一个 action token
```

---

## 第五部分：为什么要 shift action input

模型 forward 里有：

```python
bos = torch.full((bsz, 1), self.bins, dtype=torch.long, device=context_ids.device)
action_in = torch.cat([bos, action_ids[:, :-1]], dim=1)
```

这就是 Day2 里 causal LM 的 shift。

假设目标 action ids 是：

```text
a1 a2 a3 ... a12
```

模型输入的 action 部分是：

```text
BOS a1 a2 ... a11
```

模型输出要预测：

```text
a1 a2 a3 ... a12
```

为什么不能把完整 action_ids 直接输入再预测自己？

因为那样模型在预测 a5 时，输入里已经有 a5 了。

这叫信息泄漏。

所以自回归训练一定要 shift。

这点你必须和 Day2 连接起来。

文本 LM 的 labels 也会 shift。

Action LM 也是同一个原理。

---

## 第六部分：context embedding 和 action embedding 为什么分开

模型里有两个 embedding：

```python
self.context_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
self.action_embedding = nn.Embedding(bins + 1, hidden_dim)
```

context token 来自驾驶状态 vocab。

action token 来自 action bins。

它们不是同一个词表。

所以用两个 embedding 更清楚。

`bins + 1` 是因为 action token 里除了 0 到 bins-1，还需要一个 BOS action token。

然后拼接：

```python
tokens = torch.cat([context_embedding(context_ids), action_embedding(action_in)], dim=1)
```

得到完整序列：

```text
[context tokens, previous action tokens]
```

shape 是：

```text
[B, CONTEXT_LEN + ACTION_DIM, C]
```

这里 ACTION_DIM 是 12。

---

## 第七部分：causal mask 为什么重要

代码里创建 mask：

```python
mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1]), diagonal=1).bool()
```

这是上三角 mask。

它阻止当前位置看未来位置。

如果没有这个 mask，模型在训练时可能看到后面的 action token。

那它就不是在学习预测，而是在偷看答案。

你要特别注意：

context tokens 可以被 action tokens 看到。

因为动作本来就应该基于 observation。

但是 action token 不能看到未来 action token。

所以这个序列的语义是：

```text
先给 context，再自回归生成 action tokens。
```

最后模型只取 action 部分的 hidden：

```python
logits = self.head(hidden[:, -ACTION_DIM:])
```

输出：

```text
action_logits: [B, 12, bins]
```

每个 action 位置都是一个分类问题。

loss 是：

```python
cross_entropy(logits.reshape(-1, bins), action_ids.reshape(-1))
```

---

## 第八部分：和 ACT continuous 路线对比

现在把 Day6 和 Day7 对比。

Day6 ACT-style：

```text
hidden -> continuous action
loss = MSE
```

Day7 OpenVLA-style：

```text
continuous action -> action ids
context + previous action ids -> logits
loss = CrossEntropy
```

连续回归的优点是直接输出数值，没有量化误差。

缺点是它不像语言模型的 token 生成。

action token 的优点是可以复用 LLM 的生成范式。

缺点是离散化会损失精度，而且 bins 设计会影响效果。

所以不要简单说谁更好。

你要知道它们是两种建模选择。

真实项目里选哪种，取决于任务、数据、控制精度、模型结构和部署需求。

---

## 收尾

今天我们讲了 OpenVLA-style action token。

你现在应该能讲清楚：

```text
continuous trajectory [12]
-> discretize into action token ids [12]
-> context tokens + shifted action tokens
-> causal transformer
-> action logits [B, 12, bins]
-> CrossEntropy loss
```

今天最重要的一句话是：

```text
action tokenization 把连续控制问题转换成离散 token 生成问题。
```

明天 Day8，我们要把前几天真正接起来。

Day4-Day5 是 VLM hidden state。

Day6 是 continuous action head。

Day7 是 action token。

Day8 要做的是：

```text
visual tokens + state tokens + latent tokens -> hidden -> trajectory
```

也就是 VLM hidden state 接 action head。
