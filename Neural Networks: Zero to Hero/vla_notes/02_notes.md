# 02 完整笔记｜makemore：语言模型、bigram、概率和 next-token prediction

## 0. 这节课到底要学会什么

这节课从 makemore 开始，目标是让你理解语言模型最小闭环。

核心句子：

```text
语言模型不是直接“理解语言”，而是在给定上下文时预测下一个 token 的概率分布。
```

最小链路：

```text
原始文本
-> tokenizer / 字符表
-> token id 序列
-> 构造输入和目标
-> 模型输出 logits
-> softmax 得到概率
-> 用真实 next token 计算 loss
-> 训练参数
-> 采样生成
```

这节看起来是在生成名字，但它是 GPT、action-token VLA、轨迹离散化建模的起点。

## 1. makemore 是什么

makemore 的任务：给它一批名字，训练一个模型生成新的、像名字的字符串。

训练数据类似：

```text
emma
olivia
ava
isabella
...
```

模型不应该只是记住训练集，而应该学到：

```text
哪些字符常出现在开头
哪些字符组合常见
哪些字符后面容易接哪些字符
名字通常如何结束
```

这就是序列统计。

## 2. 字符级建模

课程一开始不用复杂 tokenizer，而是把每个字符当作 token。

假设字符集：

```text
a b c ... z .
```

其中 `.` 是特殊 token，表示开始或结束。

建立映射：

```text
stoi: string to integer
itos: integer to string
```

例如：

```text
stoi['a'] = 1
stoi['b'] = 2
...
stoi['.'] = 0
```

模型看到的是整数 id，不是字符本身。

## 3. 为什么需要特殊 token

名字 `emma` 可以扩展为：

```text
.emma.
```

第一个 `.` 表示“现在要开始生成名字”；最后一个 `.` 表示“名字结束”。

这样可以构造训练样本：

```text
. -> e
e -> m
m -> m
m -> a
a -> .
```

没有结束 token，模型不知道什么时候停止。没有开始 token，模型不知道第一个字符的分布。

在 GPT/VLA 里也类似：

```text
BOS: beginning of sequence
EOS: end of sequence
PAD: padding
IMAGE: 图像占位
ACTION_BOS: 动作序列开始
```

特殊 token 是模型协议的一部分。

## 4. bigram 模型

bigram 的假设：

```text
下一个字符只依赖当前字符
P(x_{t+1} | x_t)
```

例如，只根据当前是 `m`，预测下一个字符是什么。

它不看更早的上下文，所以：

```text
P(a | em) 和 P(a | xm)
```

在 bigram 里都被简化成：

```text
P(a | m)
```

这很弱，但非常适合作为语言模型入门。

## 5. 计数表

用一个二维矩阵 `N` 统计字符转移。

如果 vocab size 是 27，那么：

```text
N.shape = [27, 27]
N[i, j] = 字符 i 后面接字符 j 的次数
```

对所有名字遍历相邻字符对：

```text
for ch1, ch2 in zip(name, name[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    N[ix1, ix2] += 1
```

这个 `N` 就是 bigram 的原始统计。

## 6. 从 count 到 probability

计数不是概率。每一行要归一化：

```text
P = N / N.sum(dim=1, keepdim=True)
```

这样：

```text
P[i, j] = 当前字符为 i 时，下一个字符为 j 的概率
每一行加起来 = 1
```

注意 `keepdim=True` 很重要。没有它，广播时维度可能不对，得到错误归一化。

这是课程里很重要的工程习惯：

```text
永远盯住 shape。
```

## 7. smoothing

如果某个字符组合在训练集中从未出现，计数为 0，对应概率为 0。

问题是 log likelihood 里会出现：

```text
log(0) = -inf
```

所以通常加一点平滑：

```text
N = N + 1
```

这叫 add-one smoothing。它避免模型对未见过的转移给出绝对 0 概率。

对大模型来说，这对应一个更一般的思想：模型不要对没见过的情况完全崩掉，要有泛化和鲁棒性。

## 8. 采样生成

生成名字的过程：

```text
ix = stoi['.']
while True:
    p = P[ix]
    ix_next = sample(p)
    if ix_next == stoi['.']:
        break
    输出 itos[ix_next]
    ix = ix_next
```

关键点：不是每次取概率最大字符，而是按分布采样。

如果总取最大值，生成会非常单调；采样能产生多样性。

这和 LLM generation 一样：

```text
logits -> softmax -> sample next token
```

## 9. likelihood

模型好不好，可以看训练数据在模型下的概率。

对于一个名字 `emma`：

```text
P(. -> e) * P(e -> m) * P(m -> m) * P(m -> a) * P(a -> .)
```

这就是 likelihood。

希望训练数据越可能越好，所以要最大化 likelihood。

但很多概率相乘会变得非常小，数值不稳定，所以取 log：

```text
log_likelihood = sum(log p_i)
```

最大化 log likelihood 等价于最大化 likelihood。

训练时通常最小化 negative log likelihood：

```text
loss = -mean(log p_i)
```

## 10. NLL 的直觉

如果真实 next token 的概率是：

```text
p = 1.0   -> -log(p) = 0
p = 0.1   -> -log(p) ≈ 2.30
p = 0.01  -> -log(p) ≈ 4.61
```

所以：

```text
真实 token 概率越高，loss 越低
真实 token 概率越低，loss 越高
```

这就是语言模型训练的核心。

## 11. 神经网络版 bigram

计数表版 bigram 没有“训练参数”的感觉。神经网络版引入权重矩阵 `W`。

输入字符 id 先变成 one-hot：

```text
xenc.shape = [num_examples, vocab_size]
```

然后：

```text
logits = xenc @ W
```

如果：

```text
xenc.shape = [N, 27]
W.shape = [27, 27]
```

那么：

```text
logits.shape = [N, 27]
```

每一行是当前样本对 27 个下一个字符的打分。

## 12. logits、counts、probabilities

课程会把 logits 解释成类似 log-counts。

流程：

```text
logits = xenc @ W
counts = logits.exp()
probs = counts / counts.sum(dim=1, keepdim=True)
```

这其实就是 softmax：

```text
probs = softmax(logits)
```

logits 可以是任意实数；softmax 把它们变成概率分布。

## 13. 神经网络训练目标

真实 next token 是 `ys`。

模型给真实类别的概率：

```text
probs[range(N), ys]
```

loss：

```text
loss = -probs[range(N), ys].log().mean()
```

然后：

```text
W.grad = dloss/dW
W.data += -lr * W.grad
```

训练后的 `W` 会学到类似计数表的转移概率。

## 14. 正则化的早期影子

如果对 `W` 加惩罚：

```text
loss = nll + lambda * (W**2).mean()
```

就会鼓励权重不要太大。权重太大时，softmax 会过度尖锐，模型过度自信。

这就是 weight decay / L2 regularization 的雏形。

## 15. bigram 的根本局限

bigram 只看一个字符，所以它不能建模：

```text
更长拼写模式
名字整体结构
开头和结尾的依赖
元音辅音节奏
跨多个字符的统计规律
```

要解决，就要看更长上下文。下一节 MLP 就会从：

```text
一个字符 -> 下一个字符
```

升级为：

```text
多个字符上下文 -> 下一个字符
```

## 16. 和 GPT 的关系

GPT 不是 bigram，但训练目标相同：

```text
给定前面的 token，预测下一个 token
```

bigram：

```text
P(x_t | x_{t-1})
```

GPT：

```text
P(x_t | x_0, x_1, ..., x_{t-1})
```

差别在于上下文长度和模型能力，不在训练目标。

## 17. 和 action-token VLA 的关系

如果把动作离散化：

```text
steering/throttle/waypoints -> action token ids
```

就可以训练：

```text
P(action_t | image tokens, text command, ego state, previous actions)
```

这和语言模型非常像。

字符级 bigram：

```text
当前字符 -> 下一个字符
```

动作 token VLA：

```text
场景上下文 + 历史动作 token -> 下一个动作 token
```

本节课支撑的是：为什么“动作可以被语言模型化”。

## 18. 和 continuous-control VLA 的关系

即使你不走 action token，而是直接输出连续轨迹：

```text
hidden -> [x1, y1, x2, y2, ...]
```

你也需要理解本节，因为 VLA backbone 往往来自 LLM/VLM。语言部分仍然是 token 模型，prompt、instruction、reasoning、command 都依赖 next-token prediction 的训练传统。

## 19. 复习自测

你应该能回答：

1. 为什么文本要先转成 token id？
2. `.` 在 makemore 里为什么既能表示开始也能表示结束？
3. bigram 模型的条件概率是什么？
4. 计数表如何变成概率表？
5. 为什么归一化时要关注 `keepdim=True`？
6. 为什么要用 log likelihood？
7. NLL 为什么适合作为 loss？
8. 神经网络版 bigram 的 `W` 学到了什么？
9. logits 和 probabilities 的区别是什么？
10. 为什么采样比 argmax 更有生成多样性？
11. bigram 为什么不够？
12. action-token VLA 和字符语言模型有什么共同点？
