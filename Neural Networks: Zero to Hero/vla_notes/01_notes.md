# 01 完整笔记｜micrograd：从标量计算图到神经网络训练闭环

## 0. 这节课到底要学会什么

这节不是为了学一个叫 micrograd 的小库，而是为了把深度学习最底层的训练机制看透。

你要能把下面这条链路讲清楚：

```text
输入数据
-> forward 计算预测
-> loss 衡量预测错多少
-> backward 计算每个参数对 loss 的影响
-> optimizer 根据梯度更新参数
-> 再 forward，loss 逐渐下降
```

如果未来你做 VLA，不管模型是 CNN、ViT、LLM、VLM、trajectory head、action query、latent slot，训练本质都还是这条链路。复杂模型只是把 `forward` 变复杂，把 `loss` 变多，把数据模态变多。

## 1. 为什么要从标量开始

PyTorch 里我们平时处理的是 Tensor，但 Tensor autograd 的本质可以从标量计算图理解。

一个神经网络里的每个参数，比如一个权重 `w`，都会参与很多数学运算：

```text
w -> 乘法 -> 加法 -> 激活函数 -> 下一层 -> ... -> loss
```

我们训练时要问的问题是：

```text
如果 w 稍微变大一点，loss 会变大还是变小？变化多少？
```

这就是梯度：

```text
dloss / dw
```

micrograd 用标量把这件事讲明白。标量够小，能看见每一步；理解后再回到 Tensor，就不会把 `loss.backward()` 当黑箱。

## 2. `Value` 是什么

micrograd 的核心对象是 `Value`。它不是普通数字，而是一个带计算历史的数字。

一个 `Value` 至少要保存：

```text
data: 当前数值
_grad: 当前节点对最终 loss 的梯度
_prev: 这个节点由哪些节点计算而来
_op: 当前节点对应什么操作
_backward: 如何把梯度传给父节点
```

直觉上：

```text
普通 float: 只知道自己是多少
Value: 知道自己是多少，也知道自己怎么来的，还能参与反向传播
```

比如：

```text
a = Value(2.0)
b = Value(3.0)
c = a * b
```

此时 `c.data = 6.0`，但更重要的是 `c` 记住了：

```text
c 是由 a 和 b 通过乘法得到的
```

这就构成了计算图的一条边。

## 3. forward 不只是算数，也是在建图

每次写：

```text
c = a + b
d = c * e
L = tanh(d)
```

forward 做了两件事：

```text
1. 算出每个中间节点的数值
2. 记录每个中间节点从哪里来
```

最后得到一个有向无环图：

```text
a ----\
       + -> c ----\
b ----/           * -> d -> tanh -> L
e ----------------/
```

训练要做的是从最后的 `L` 或 `loss` 开始，把梯度一路传回所有叶子节点。

## 4. 梯度的含义

如果：

```text
L = f(a)
```

那么 `a.grad` 表示：

```text
dL / da
```

也就是 `a` 变化一点点时，`L` 会变化多少。

常见直觉：

```text
grad > 0: a 增大会让 loss 增大，所以训练时应该减小 a
grad < 0: a 增大会让 loss 减小，所以训练时应该增大 a
grad = 0: 当前局部位置，a 对 loss 没有一阶影响
```

参数更新公式：

```text
p.data += -learning_rate * p.grad
```

这就是梯度下降。

## 5. 链式法则是 backward 的全部

假设：

```text
c = a * b
L = f(c)
```

我们已知上游梯度：

```text
dL / dc
```

局部导数是：

```text
dc / da = b
dc / db = a
```

链式法则：

```text
dL / da = dL / dc * dc / da
dL / db = dL / dc * dc / db
```

所以乘法节点的 `_backward` 可以写成：

```text
a.grad += b.data * c.grad
b.grad += a.data * c.grad
```

加法节点：

```text
c = a + b
dc/da = 1
dc/db = 1

a.grad += 1 * c.grad
b.grad += 1 * c.grad
```

`backward` 没有魔法，就是每个局部操作知道自己的局部导数，然后把上游梯度乘进去。

## 6. 为什么梯度要 `+=`，不是 `=`

这是这节课最容易漏掉但最关键的点之一。

如果一个变量被使用了多次：

```text
b = a + a
```

数学上：

```text
b = 2a
db/da = 2
```

但计算图里 `a` 会沿两条边传到 `b`。如果 backward 时用赋值 `=`，第二条边会覆盖第一条边，梯度就错了。

正确方式是累加：

```text
a.grad += contribution_from_path_1
a.grad += contribution_from_path_2
```

PyTorch 也是这样。每次 `loss.backward()` 都会把梯度累积到参数的 `.grad` 上，所以训练循环里必须：

```text
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

如果忘了 `zero_grad()`，当前 step 的梯度会叠加上前面 step 的梯度，训练就会变味。

## 7. 拓扑排序为什么必要

反向传播必须按依赖顺序来。

如果：

```text
a, b -> c -> d -> L
```

要算 `a.grad`，必须先知道 `c.grad`；要知道 `c.grad`，必须先从 `d` 和 `L` 传回来。

所以 backward 顺序应该是：

```text
L -> d -> c -> a,b
```

micrograd 做法：

```text
1. 从最终节点出发 DFS
2. 收集所有节点形成拓扑序
3. 反向遍历拓扑序，调用每个节点的 _backward
```

这和 PyTorch autograd 的核心思想一致。

## 8. 激活函数为什么也只是一个节点

比如 `tanh`：

```text
t = tanh(x)
```

它的局部导数：

```text
dt/dx = 1 - tanh(x)^2 = 1 - t^2
```

所以 backward：

```text
x.grad += (1 - t.data**2) * t.grad
```

这说明 ReLU、sigmoid、GELU、softmax、LayerNorm 等都可以理解成计算图里的操作。只要能写出局部导数或由 autograd 组合出来，就能反向传播。

## 9. 从一个 neuron 到 MLP

一个 neuron 的 forward：

```text
n = w1*x1 + w2*x2 + ... + wk*xk + b
out = tanh(n)
```

参数是：

```text
w1, w2, ..., wk, b
```

一层 layer 是多个 neuron 并排：

```text
x -> [neuron_1, neuron_2, ..., neuron_m]
```

MLP 是多层 layer 串起来：

```text
input -> layer1 -> layer2 -> layer3 -> output
```

micrograd 里每个权重都是 `Value`，最后 loss 对每个权重都有梯度。

## 10. loss 怎么定义

课程里会用一个简单的监督学习例子：给定输入 `x`，希望模型输出接近目标 `y`。

常见 MSE loss：

```text
loss = sum((y_pred - y_true)^2)
```

如果预测错得多，loss 大；预测接近，loss 小。

反向传播会告诉每个参数应该怎么动，才能让这个 loss 下降。

## 11. 训练循环的完整骨架

你以后要把这个骨架背到肌肉里：

```text
for step in range(num_steps):
    # forward
    y_pred = model(x)
    loss = criterion(y_pred, y)

    # backward
    for p in parameters:
        p.grad = 0
    loss.backward()

    # update
    for p in parameters:
        p.data += -lr * p.grad
```

PyTorch 版本：

```python
for batch in loader:
    optimizer.zero_grad()
    pred = model(batch)
    loss = loss_fn(pred, target)
    loss.backward()
    optimizer.step()
```

这就是所有大模型训练脚本最里面的心跳。

## 12. micrograd 和 PyTorch 的对应关系

```text
micrograd Value.data       <-> torch.Tensor 数值
micrograd Value.grad       <-> tensor.grad
Value._prev / _backward    <-> PyTorch autograd graph
loss.backward()            <-> 反向遍历计算图
手动 p.data += -lr*p.grad  <-> optimizer.step()
```

区别是：

```text
micrograd: 标量级别，教学清晰
PyTorch: Tensor 级别，向量化/GPU/高性能
```

但数学机制是同一个。

## 13. 常见坑

### 13.1 忘记清零梯度

现象：loss 乱跳，更新过大。

原因：`.grad` 默认累积。

### 13.2 原地修改参数时被 autograd 追踪

PyTorch 中通常在 optimizer 内部或 `torch.no_grad()` 下更新参数，避免把参数更新本身记进计算图。

### 13.3 计算图断了

常见原因：

```text
.detach()
torch.no_grad()
把 tensor 转成 numpy 再转回来
错误使用 .item()
冻结了不该冻结的参数
```

断图后，loss 不会训练到前面的模块。

### 13.4 梯度为 None 和梯度为 0 不一样

```text
grad is None: 这个参数没有参与产生 loss，或没有 requires_grad
grad == 0: 参与了，但当前局部导数为 0 或抵消
```

debug 时要区分。

## 14. 对 VLA 全栈的意义

VLA 的完整链路可能是：

```text
camera image
+ lidar / BEV / map
+ language command
+ ego state
+ history trajectory
-> multimodal encoder
-> fusion Transformer
-> planning hidden / action query / latent slot
-> trajectory / control / language / world heads
-> multiple losses
```

只要训练，这些 loss 都要通过计算图回传。

你要能判断：

```text
trajectory loss 会不会训练 vision encoder？
language auxiliary loss 会不会影响 action head？
world model loss 会不会塑造 shared hidden？
冻结 LLM 后，projector 还会不会学？
detach visual features 后，vision encoder 还会不会学？
```

这些问题的答案都来自本节课。

## 15. 复习自测

你应该能不看答案回答：

1. `Value` 为什么要保存 `_prev` 和 `_backward`？
2. 为什么反向传播要从 loss 开始？
3. 加法节点和乘法节点的 backward 分别是什么？
4. 为什么梯度要累加？
5. 为什么需要拓扑排序？
6. `optimizer.zero_grad()` 为什么必须在每步训练里出现？
7. `detach()` 对训练有什么影响？
8. 一个 VLA 模型里，如果 trajectory head 学了但 vision encoder 没学，可能有哪些原因？
9. 多个 loss 共享 backbone 时，梯度如何合并？
10. 为什么说 auxiliary loss 本质是在塑造中间表示？
