# 06 完整笔记｜WaveNet：层级序列融合、模块化网络和 shape 思维

## 0. 这节课到底要学会什么

前面的 MLP 用固定长度上下文预测下一个字符，但它把所有上下文 embedding 直接 flatten，再一次性送入 hidden layer。

这节课升级方向：

```text
不要一次性压扁上下文，而是让相邻 token 逐层融合，构造更深的序列模型。
```

它会走向 WaveNet-like 架构，也会训练你理解：

```text
[B, T, C]
batch 维
time/token 维
channel/hidden 维
```

这是 Transformer/VLM/VLA 的核心 shape 语言。

## 1. 旧 MLP 的问题

旧模型：

```text
X:        [B, T]
emb:      [B, T, E]
emb_flat: [B, T*E]
hidden:   [B, H]
logits:   [B, V]
```

问题：

```text
1. flatten 太早，token/time 结构被抹掉
2. 上下文长度 T 增加时，第一层参数变多
3. 所有信息一次性混合，没有层级结构
4. 不能自然表达局部组合到全局组合的过程
```

语言、音频、轨迹都有局部结构。相邻 token 往往先形成局部 pattern，再逐层组合成更大 pattern。

## 2. WaveNet 的基本思想

WaveNet 原本用于音频生成，本质也是自回归序列建模：

```text
过去的音频 samples -> 预测下一个 sample
```

它用层级/卷积式结构逐步扩大感受野。

在字符模型里可以类比：

```text
先把相邻字符 pair 融合
再把 pair 融合成更长片段
最后得到整个上下文的表示
```

例如 8 个 token：

```text
第 0 层: 8 个 token
第 1 层: 4 个 pair 表示
第 2 层: 2 个 4-token 表示
第 3 层: 1 个 8-token 表示
```

这比直接 flatten 更有结构。

## 3. 感受野 receptive field

感受野表示某个 hidden 表示能看到多少输入 token。

在层级融合中：

```text
第一层每个表示看 2 个 token
第二层每个表示看 4 个 token
第三层每个表示看 8 个 token
```

深度增加，感受野扩大。

Transformer self-attention 则更直接：每个 token 可以 attend 到很多 token。但二者都在解决同一个问题：让序列位置获得上下文信息。

## 4. 为什么要模块化

课程会把之前手写的层封装成类似 PyTorch 的模块：

```text
Linear
BatchNorm1d
Tanh
Embedding
Flatten
Sequential
```

每个模块都有：

```text
forward
parameters
training/eval 状态
```

这让模型搭建更清晰，也让你理解 PyTorch `nn.Module` 背后不是魔法。

## 5. `Embedding` 模块

输入：

```text
X: [B, T]
```

输出：

```text
emb: [B, T, E]
```

它本质是查表：

```text
weight.shape = [V, E]
out = weight[X]
```

VLA 类比：

```text
text token embedding
action token embedding
special token embedding
```

## 6. `Flatten` 的问题

普通 Flatten：

```text
[B, T, E] -> [B, T*E]
```

它把所有 token 直接压成一个向量。

WaveNet-like 需要的是局部 flatten，例如把相邻 2 个 token 合并：

```text
[B, T, E] -> [B, T/2, 2E]
```

这叫 `FlattenConsecutive(2)`。

## 7. `FlattenConsecutive` 的 shape

如果：

```text
x.shape = [B, 8, C]
```

做 consecutive=2：

```text
x.view(B, 4, 2*C)
```

得到：

```text
[B, 4, 2C]
```

意思是每相邻两个 token 的 channel 拼起来，token 数减半，channel 数翻倍。

再接 Linear：

```text
Linear(2C, H)
```

把每个 pair 表示映射回 hidden dim。

## 8. 层级结构示例

一个 WaveNet-like 模型可以是：

```text
Embedding(V, C)
FlattenConsecutive(2)
Linear(2C, H)
BatchNorm1d(H)
Tanh()
FlattenConsecutive(2)
Linear(2H, H)
BatchNorm1d(H)
Tanh()
FlattenConsecutive(2)
Linear(2H, H)
BatchNorm1d(H)
Tanh()
Linear(H, V)
```

如果输入上下文长度是 8：

```text
[B, 8] -> [B, 8, C]
-> [B, 4, 2C]
-> [B, 4, H]
-> [B, 2, 2H]
-> [B, 2, H]
-> [B, 1, 2H]
-> [B, 1, H]
-> [B, V]
```

这就逐层融合了 8 个 token。

## 9. BatchNorm1d 的维度问题

PyTorch 的 BatchNorm1d 通常期望：

```text
[N, C]
```

或：

```text
[N, C, L]
```

但我们的序列中常见：

```text
[B, T, C]
```

课程里会处理这个问题：对最后一维 channel 做归一化，同时把前面的 batch/time 维看成 batch-like 维。

也就是统计：

```text
mean over B and T
var over B and T
```

而不是只 over B。

## 10. 多个 batch-like 维度

一个重要思想：

```text
有些层只关心最后一维 C，前面的所有维度都可以看成 batch 维。
```

例如 Linear：

```text
x: [B, T, C]
W: [C, H]
out: [B, T, H]
```

它对每个 `[C]` 向量独立应用同一个线性变换。

这也是 Transformer FeedForward 的工作方式：对每个 token position 独立应用 MLP。

## 11. 为什么 squeeze 要小心

如果某个维度大小为 1，`squeeze()` 可能把它去掉。

例如：

```text
[B, 1, H] -> [B, H]
```

有时这是想要的，有时会破坏后续层预期。

工程习惯：明确指定 squeeze 哪个维度，或者在模块里谨慎处理 shape。

## 12. 训练更深网络时看什么

这节继续沿用第 4 节的诊断：

```text
activation mean/std
activation saturation
gradient mean/std
parameter update ratio
train/dev loss
```

更深的网络更需要这些诊断。

## 13. WaveNet 和 causality

真正 WaveNet 用 causal convolution，保证预测当前位置时不能看未来。

makemore 这里用固定上下文预测下一个字符，也天然只使用过去上下文。

后面 GPT 用 causal mask 实现同样约束。

对 action-token VLA 也一样：

```text
预测 action_t 时不能看未来 action_{t+1}
```

## 14. 和 Transformer 的关系

WaveNet-like 层级融合：

```text
局部 -> 更大局部 -> 全局
```

Transformer attention：

```text
每个 token 直接根据 attention 权重读取其他 token
```

二者不同，但共同关注：

```text
token 之间如何交换信息
上下文如何进入每个位置的表示
```

## 15. 和 VLA 全栈的连接

VLA 输入不是单一序列，可能包括：

```text
图像 patch tokens
BEV/grid tokens
map polyline tokens
text instruction tokens
ego/history state tokens
latent/action query tokens
```

你要关心的不是“它叫 WaveNet 还是 Transformer”，而是：

```text
这些 token 如何融合？
局部空间信息如何扩大到全局场景理解？
历史轨迹如何影响未来 action？
动作输出槽位如何 attend 到 observation？
```

## 16. 对 trajectory/action chunk 的启发

动作不是单个标量。自动驾驶常预测：

```text
未来 N 个 waypoints
或未来 N 步 control
```

这也是序列。

可以：

```text
一次性 flatten 输出所有点
用 action queries 每个未来步一个槽位
用自回归 action tokens 逐步生成
用层级结构先预测粗轨迹再细化
```

WaveNet 这节让你意识到：输出和输入都可能有序列结构，不要过早 flatten。

## 17. 常见误区

### 17.1 把 `[B, T, C]` 维度混掉

VLA 中最常见 bug 就是维度含义混乱。一定要知道每个维度代表什么。

### 17.2 以为 flatten 越早越简单越好

简单是简单，但可能丢掉结构。

### 17.3 不理解模块复用

Linear 可以作用在 `[B, C]`，也可以作用在 `[B, T, C]` 的最后一维。这是很多高级模型的基础。

### 17.4 忽略 causal 约束

序列预测任务里，训练时偷看未来会导致评估崩掉。

## 18. 复习自测

1. 旧 MLP 为什么要 flatten？它有什么问题？
2. WaveNet-like 层级融合的核心思想是什么？
3. receptive field 是什么？
4. `FlattenConsecutive(2)` 如何改变 shape？
5. `[B, T, C]` 中三个维度分别是什么？
6. Linear 如何作用在三维输入上？
7. BatchNorm1d 在序列输入上为什么要小心？
8. WaveNet 和 Transformer 都在解决什么共同问题？
9. VLA 中哪些输入可以看成 token sequence？
10. 预测 action chunk 时，为什么不应该只把它看成普通回归向量？
