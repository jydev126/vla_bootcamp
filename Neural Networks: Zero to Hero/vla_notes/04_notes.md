# 04 完整笔记｜Activations & Gradients, BatchNorm：训练稳定性和网络诊断

## 0. 这节课到底要学会什么

前三节已经能训练一个 MLP 语言模型了。但这节课要告诉你：

```text
模型结构写对 ≠ 模型训练得好
```

神经网络训练是否顺利，很大程度取决于：

```text
初始化
激活分布
梯度分布
归一化
学习率
参数尺度
```

这节课的核心能力是：你不能只看 loss，要能看懂网络内部的 activations 和 gradients。

对 VLA 来说，这非常关键。VLA 是多模态、多模块、多 loss 系统，训练失败经常不是因为“架构不先进”，而是某个 projector、adapter、action head、latent slot、normalization 或 loss scale 把梯度弄坏了。

## 1. 为什么还停在 MLP，不急着上 RNN/Transformer

课程开头说，本来可以继续到 RNN、GRU、LSTM 等更复杂序列模型，但必须先理解激活和梯度。

原因：

```text
复杂架构的很多历史演化，都是为了让梯度更好流动，让激活分布更稳定。
```

如果你不理解 activation saturation、gradient vanishing/exploding、initialization scale，后面看到 residual、LayerNorm、BatchNorm、RMSNorm、attention scaling，只会背名词。

## 2. 从上一节 MLP 的训练现象开始

上一节 MLP 能训练，但 loss 曲线里有一些异常：

```text
初始 loss 可能很高
训练前几步 loss 快速下降
后面下降变慢
train/dev loss 差距提示容量或过拟合问题
```

这节会问：

```text
这些现象是不是模型内部数值状态导致的？
```

答案是：很多时候是。

## 3. 初始 loss 应该是多少

这是非常重要的 sanity check。

如果 vocab size 是 27，模型初始化时还什么都没学，理想情况下对 27 个字符应该差不多均匀预测。

均匀概率：

```text
p = 1 / 27
```

交叉熵：

```text
loss = -log(1/27) ≈ 3.296
```

所以如果一开始 loss 远大于 3.3，比如 20 多，就说明模型初始化状态很糟。

## 4. 初始 loss 过高意味着什么

初始 loss 过高通常说明 logits 太极端。

如果模型随机初始化后，对错误类别给了非常高概率，对真实类别给了极低概率，NLL 会非常大。

例子：

```text
真实类别概率 p_true = 1e-10
loss = -log(1e-10) ≈ 23
```

这并不是模型“还没学所以正常”，而是初始化让模型一开始过度自信。

健康初始化应该让 logits 接近 0，使 softmax 接近均匀分布。

## 5. 如何修正输出层初始化

如果最后一层权重太大，logits 方差大，softmax 会非常尖锐。

可以把最后一层权重乘小：

```text
W2 *= 0.01
b2 = 0
```

这样初始 logits 接近 0，初始 loss 接近理论均匀值。

这个思想对大模型也成立：很多模型会专门缩小输出层或 residual projection 的初始化，避免一开始数值爆炸。

## 6. hidden activation 的问题

修好初始 logits 后，还要看 hidden layer 的 activation。

上一节 MLP 用：

```text
h = tanh(hpreact)
```

`tanh` 输出范围：

```text
[-1, 1]
```

如果 `hpreact` 绝对值很大，`tanh` 会饱和到 -1 或 1。

## 7. tanh 饱和为什么危险

`tanh` 的导数：

```text
d tanh(x) / dx = 1 - tanh(x)^2
```

如果：

```text
tanh(x) ≈ 1 或 -1
```

那么：

```text
1 - tanh(x)^2 ≈ 0
```

这意味着梯度几乎传不过去。

直觉：

```text
神经元输出已经贴到边界，再怎么微调输入，输出几乎不变
```

所以前面的参数收不到有效梯度。

## 8. 如何观察 activation

课程会画 histogram 或统计：

```text
h.mean()
h.std()
h.abs() > 0.99 的比例
```

如果大量 activation 接近 ±1，说明 tanh 饱和。

可以把饱和神经元理解成训练时“暂时死掉”的通路。

## 9. pre-activation 的尺度

`tanh` 饱和来自 `hpreact` 太大。

```text
hpreact = emb_flat @ W1 + b1
```

如果 `W1` 初始化太大，或者输入 embedding 尺度太大，`hpreact` 方差会大。

修正方式：

```text
W1 *= smaller_scale
```

让 `hpreact` 分布更集中在 0 附近。

`tanh` 在 0 附近近似线性，梯度更健康。

## 10. 但也不能太小

如果权重太小：

```text
hpreact ≈ 0
h ≈ 0
```

虽然不饱和，但信号也很弱。后续层看到的信息太小，表达能力变差。

初始化的目标不是“越小越好”，而是让每层 activation 有合适方差。

## 11. fan-in 和初始化尺度

对于线性层：

```text
y = x @ W
```

如果输入维度是 `fan_in`，每个输出是 `fan_in` 个随机项相加。

如果每项方差不控制，输出方差会随 `fan_in` 增大。

所以常见初始化会按 fan-in 缩放：

```text
W ~ N(0, 1/sqrt(fan_in))
```

对于 ReLU/tanh 等不同激活，还会有 gain 系数。

## 12. Kaiming/He 初始化的直觉

Kaiming 初始化的目标：

```text
让信号经过线性层 + 非线性后，方差不要逐层爆炸或消失
```

不同激活函数需要不同 gain。

例如 ReLU 会把一半负值截断，所以方差变化不同，需要对应缩放。

这节不要求死背公式，但要知道：

```text
初始化不是随便乘一个随机数，而是在控制激活和梯度统计。
```

## 13. 为什么要看 gradient histogram

forward activation 健康，不代表 backward gradient 健康。

要看：

```text
每层 activation 的均值/标准差/饱和比例
每层 gradient 的均值/标准差
参数梯度和参数值的比例
```

如果某层梯度特别小，前面层几乎不学。

如果某层梯度特别大，训练可能不稳定。

## 14. update-to-data ratio

一个很实用的诊断：

```text
参数更新量 / 参数本身尺度
```

例如：

```text
update = -lr * grad
ratio = update.std() / param.std()
```

如果 ratio 太大，说明每一步把参数改得太猛。

如果 ratio 太小，说明训练很慢。

大模型训练也会用类似思想观察梯度范数、更新范数。

## 15. BatchNorm 的动机

如果我们想让每层 pre-activation 保持均值 0、方差 1，能不能直接标准化？

BatchNorm 做的就是：

```text
hpreact_norm = (hpreact - mean) / std
```

其中 mean/std 从当前 batch 统计。

这样可以让激活分布更稳定。

## 16. BatchNorm 公式

对 batch 中某层 pre-activation：

```text
mean = hpreact.mean(dim=0)
var = hpreact.var(dim=0)
hat = (hpreact - mean) / sqrt(var + eps)
out = gamma * hat + beta
```

`gamma` 和 `beta` 是可训练参数。

为什么需要它们？

如果永远强制均值 0、方差 1，可能限制模型表达。`gamma/beta` 允许模型学习合适的尺度和平移。

## 17. BatchNorm 的训练/推理差异

训练时：

```text
使用当前 batch 的 mean/var
```

推理时：

```text
不能依赖一个 batch 的统计
使用训练过程中累计的 running mean/var
```

所以 BatchNorm module 需要 train/eval 模式。

这是你必须掌握的工程点：

```text
model.train()
model.eval()
```

不仅影响 Dropout，也影响 BatchNorm。

## 18. BatchNorm 带来的副作用

BatchNorm 让 batch 内样本相互耦合：一个样本的 normalized value 依赖同 batch 其他样本。

这有时像正则化，但也带来问题：

```text
小 batch 统计不稳定
训练/推理不一致
序列模型和变长输入中使用不如 LayerNorm 自然
```

Transformer 更常用 LayerNorm/RMSNorm，因为它们不依赖 batch 维统计。

## 19. 为什么 BatchNorm 后 bias 可能没用

如果：

```text
hpreact = x @ W + b
```

然后 BatchNorm 会减掉 batch mean。

这个 mean 包含了 bias，所以 bias 的平移效果会被标准化抵消。

因此 Linear 后接 BatchNorm 时，前面 Linear 的 bias 常常可以省掉。

## 20. deeper network 的诊断

课程后面会堆更多层，然后观察：

```text
每层 activation histogram
每层 gradient histogram
每层参数 gradient:data ratio
```

理想情况：

```text
activation 不爆炸、不消失
梯度在各层比较均衡
没有大量 tanh 饱和
参数更新比例合理
```

如果越深越坏，说明初始化/归一化/激活设计不合适。

## 21. 这节课和后续架构的历史关系

很多现代架构设计都在解决这类问题：

```text
Residual connection: 让梯度有更直接路径
LayerNorm/RMSNorm: 稳定 hidden 分布
Attention scaling: 防止 dot product 太大
GELU/ReLU: 改善激活性质
Pre-LN Transformer: 深层训练更稳定
初始化缩放: 控制 residual 累积
```

所以这节是理解 Transformer 训练稳定性的前置课。

## 22. 对 VLA 全栈的意义

VLA 里常见风险：

```text
vision encoder 输出尺度和 LLM hidden 不匹配
projector 初始化太大，破坏语言 hidden
latent slot 初始化不合理，抢占 attention 或学不动
action head 输出尺度和 waypoint/control label 尺度不匹配
多任务 loss 梯度量级差异太大
冻结/解冻策略导致某些模块梯度为 None
```

这些问题不是“玄学”，都可以用 activation/gradient 诊断。

## 23. VLA 中应该观察什么

训练一个 VLA toy 或真实模型时，建议记录：

```text
visual tokens mean/std
text hidden mean/std
projector output mean/std
latent/action query hidden mean/std
trajectory head input mean/std
trajectory prediction scale
各模块 gradient norm
各 loss 的数值和梯度贡献
```

如果 action loss 不降，不要只调模型结构。先看数值健康。

## 24. 常见误区

### 24.1 只看 loss

loss 是最终结果，不告诉你内部哪里坏。

### 24.2 初始化随便写

小模型可能能忍，大模型/多模态模型会放大问题。

### 24.3 以为 normalization 总是无害

BatchNorm、LayerNorm、RMSNorm 都会改变表示分布。位置不对可能影响训练或推理。

### 24.4 忽略 label scale

VLA 输出连续控制时，label scale 很重要。waypoint 单位、归一化、坐标系都会影响 loss 和梯度。

## 25. 复习自测

1. 初始 loss 为什么应该接近 `-log(1/vocab_size)`？
2. logits 初始化太大为什么会导致初始 loss 异常高？
3. `tanh` 饱和为什么会让梯度消失？
4. 怎么统计一个 hidden layer 的饱和比例？
5. fan-in 初始化的核心直觉是什么？
6. BatchNorm 标准化的是哪一维？
7. BatchNorm 为什么需要 `gamma` 和 `beta`？
8. BatchNorm 训练和推理有什么不同？
9. 为什么 Transformer 常用 LayerNorm 而不是 BatchNorm？
10. VLA 里 projector 输出尺度不匹配会造成什么问题？
11. 多任务 loss scale 不平衡时会发生什么？
12. action head 学不动时，你会先查哪些 activation/gradient？
