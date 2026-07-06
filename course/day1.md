# Day 1：PyTorch / 深度学习最小闭环上课台词

同学，今天我们不追求学完整个深度学习体系。今天只有一个目标：你要亲手跑通一个最小训练闭环。

所谓最小训练闭环，就是这几步：

先准备数据，然后把数据送进模型，模型输出预测值，再用预测值和真实值计算 loss，然后反向传播，再用 optimizer 更新参数，最后保存结果。

你今天必须把这条线背下来：

```python
pred = model(x)
loss = loss_fn(pred, y)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

以后你看 OneVL、VLA、BEV、UniAD 这些复杂训练代码，本质上都逃不开这几句。复杂模型只是把 `model(x)` 变复杂了，把 `loss` 变复杂了，把 `batch` 变复杂了，但训练主线不变。

你现在的基础是：会写代码，懂一些机器学习，但深度学习不系统。所以今天我不会一上来讲 Transformer、Attention、VLA。那些太远了。我们先把 PyTorch 训练模型的肌肉记忆建立起来。

今天学完，你要能回答五个问题：

第一，Tensor 的 shape 怎么看？

第二，`nn.Module` 的 `forward` 到底做什么？

第三，`loss.backward()` 到底做什么？

第四，`detach`、`no_grad`、`requires_grad` 有什么区别？

第五，`Dataset` 和 `DataLoader` 怎么分工？

如果这五个问题你能解释清楚，Day 1 就合格。

---

## 第一部分：先看整体，PyTorch 到底在干什么

我们先不要急着看代码。先建立一个大图。

深度学习训练模型，其实就是一个“不断调参数”的过程。

我们有一个模型：

```python
pred = model(x)
```

这里 `x` 是输入，`pred` 是模型的预测结果。

然后我们有真实答案 `y`。我们要比较 `pred` 和 `y` 差多少：

```python
loss = loss_fn(pred, y)
```

这个 `loss` 可以理解为错误程度。loss 越大，说明模型预测越差；loss 越小，说明模型越接近真实答案。

接下来我们希望模型自动调整自己的参数，让下次预测更准。

这一步分成两步：

第一步，计算梯度：

```python
loss.backward()
```

第二步，根据梯度更新参数：

```python
optimizer.step()
```

注意，这里有一个非常关键的点：

`loss.backward()` 不更新参数。

它只是计算“每个参数应该往哪个方向改，以及改多少趋势”。

真正更新参数的是：

```python
optimizer.step()
```

所以你一定要背下来：

```text
backward 负责算梯度，step 负责更新参数。
```

在每一次算新梯度之前，还要清空上一轮残留的梯度：

```python
optimizer.zero_grad()
```

所以完整顺序是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

这就是今天的核心。

---

## 第二部分：先解释 import，不要觉得 import 不重要

现在我们看第一个脚本开头：

```python
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
```

很多初学者会跳过 import，但其实这里已经出现了今天的几个核心概念。

先看：

```python
import torch
```

`torch` 是 PyTorch 的主包。你可以把它理解成 PyTorch 的基础工具箱。创建 Tensor、做数学运算、存模型、控制设备，都要用它。

比如：

```python
torch.linspace
torch.sin
torch.save
torch.manual_seed
```

这些都来自 `torch`。

再看：

```python
import torch.nn as nn
```

这里的 `nn` 是 neural network 的缩写，也就是神经网络模块。

你今天会看到：

```python
nn.Module
nn.Linear
nn.ReLU
nn.Sequential
nn.MSELoss
```

这些都是 `torch.nn` 下面的东西。

你可以这样理解：

`torch` 负责基础张量计算，`torch.nn` 负责搭神经网络。

接着看：

```python
import torch.optim as optim
```

这个 `optim` 很重要。它是 optimizer 的缩写，中文一般叫“优化器”。

你可能以前没接触过 optimizer。我们今天掌握到什么程度？

你不需要今天手推 AdamW 公式。

你只需要知道：

optimizer 的作用是根据梯度更新模型参数。

比如：

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3)
```

这句话的意思是：

“创建一个 AdamW 优化器，让它管理模型的所有可训练参数，学习率是 0.001。”

后面：

```python
optimizer.step()
```

就是让 optimizer 真正更新这些参数。

所以今天你先背：

```text
optim 是 PyTorch 的优化器模块。
optimizer 管理模型参数。
optimizer.step() 根据梯度更新参数。
```

再看：

```python
import matplotlib.pyplot as plt
```

这个是画图用的。我们今天用它保存 loss 曲线和预测曲线。

最后：

```python
from pathlib import Path
```

这个不是深度学习知识，是 Python 工程写法。它让我们更方便地处理路径，比如创建输出目录、保存图片、保存 checkpoint。

---

## 第三部分：设备 device，为什么要写 CPU / CUDA

代码里有这个函数：

```python
def choose_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            _ = torch.randn(1, device="cuda")
            return torch.device("cuda")
    except Exception as e:
        print(f"[WARN] CUDA is not usable, fallback to CPU. Reason: {e}")
    return torch.device("cpu")
```

老师这里要提醒你：今天你不需要深入 CUDA，但你要知道 PyTorch 的 Tensor 和模型可以放在 CPU 或 GPU 上。

GPU 在 PyTorch 里通常叫 CUDA 设备。

比如：

```python
device = torch.device("cuda")
```

表示用 GPU。

```python
device = torch.device("cpu")
```

表示用 CPU。

为什么代码里不直接写 CUDA？

因为你的机器上可能出现这种情况：PyTorch 看得到 CUDA，但驱动版本不匹配，真正运行会报错。

所以这里写了一个安全判断：

```python
if torch.cuda.is_available():
    _ = torch.randn(1, device="cuda")
```

这行代码创建一个很小的 Tensor，放到 CUDA 上试一下。如果能成功，说明 CUDA 真能用。如果失败，就退回 CPU。

今天这个 MLP 很小，CPU 也能跑。所以你的重点不是纠结 CUDA，而是理解：

```text
Tensor 和 model 必须在同一个 device 上。
```

所以后面你会看到：

```python
x = x.to(device)
y = y.to(device)
model = MLP().to(device)
```

这三句话很重要。

如果 `x` 在 GPU，`model` 在 CPU，PyTorch 会报错。

背下来：

```text
输入、标签、模型，必须放在同一个设备上。
```

---

## 第四部分：Tensor 和 shape

现在进入第一个关键概念：Tensor。

代码：

```python
x = torch.linspace(-10, 10, 1000).unsqueeze(1)
y = torch.sin(x) + 0.3 * x
```

我们一句一句看。

先看：

```python
torch.linspace(-10, 10, 1000)
```

这句话会在 -10 到 10 之间均匀生成 1000 个数。

如果不加 `unsqueeze(1)`，它的 shape 是：

```python
[1000]
```

也就是一个一维向量。

但是我们写神经网络时，通常希望输入是：

```python
[样本数, 特征维度]
```

所以我们加：

```python
unsqueeze(1)
```

把 shape 从：

```python
[1000]
```

变成：

```python
[1000, 1]
```

这里你必须彻底理解。

`[1000, 1]` 不是随便写的。

`1000` 表示有 1000 个样本。

`1` 表示每个样本只有一个特征，也就是一个 x 值。

如果你以后做图像，shape 可能是：

```python
[B, C, H, W]
```

如果做文本，shape 可能是：

```python
[B, L]
```

如果做轨迹预测，shape 可能是：

```python
[B, T, D]
```

但不管什么任务，第一维经常都是 batch 维，也就是一次送进模型多少个样本。

今天我们先只记住：

```python
x.shape = [1000, 1]
```

其中：

```text
1000 是样本数量。
1 是每个样本的特征维度。
```

接着看：

```python
y = torch.sin(x) + 0.3 * x
```

这是我们要拟合的真实函数：

```text
y = sin(x) + 0.3x
```

也就是说，我们自己生成训练数据，不需要外部数据集。

这个任务叫回归任务。

为什么叫回归？

因为模型输出的是连续值，不是类别。

如果我们预测“猫还是狗”，那是分类。

如果我们预测某个连续数值，比如轨迹点坐标、速度、距离、函数值，那就是回归。

今天这个任务就是最简单的一维回归。

---

## 第五部分：为什么要打印 shape

代码里有：

```python
print("x:", x.shape)
print("y:", y.shape)
```

后面还有：

```python
print("pred:", pred.shape)
```

老师这里要强调：刚开始学深度学习，最重要的习惯之一就是打印 shape。

因为深度学习代码里，很多 bug 不是语法错，而是 shape 错。

比如模型要求输入 `[B, 1]`，你给了 `[B]`，可能就报错。

再比如 loss 要求 `pred` 和 `y` shape 一样，你一个是 `[1000, 1]`，一个是 `[1000]`，可能会触发广播机制，结果训练不符合预期。

所以今天你要养成习惯：

```text
每写一段模型输入输出，都先打印 shape。
```

你今天看到：

```python
x: torch.Size([1000, 1])
y: torch.Size([1000, 1])
pred: torch.Size([1000, 1])
```

这说明：

输入是 1000 个样本，每个样本 1 维。

标签也是 1000 个样本，每个标签 1 维。

模型输出也是 1000 个样本，每个预测 1 维。

这样 loss 才能正确计算。

---

## 第六部分：nn.Module 是什么

现在看模型代码：

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
```

我们慢慢讲。

首先：

```python
class MLP(nn.Module):
```

这表示我们定义了一个模型，叫 MLP。

MLP 是 Multi-Layer Perceptron，多层感知机。

你今天可以把它理解成最基础的全连接神经网络。

为什么要继承 `nn.Module`？

因为在 PyTorch 里，所有神经网络模型都应该继承 `nn.Module`。

继承之后，PyTorch 才能帮你管理模型参数，切换 train/eval 状态，保存和加载参数。

你现在只需要背：

```text
nn.Module 是 PyTorch 里模型的基类。
自己写模型，一般都继承 nn.Module。
```

接着看：

```python
def __init__(self):
    super().__init__()
```

`__init__` 是初始化函数。

这里一般定义模型有哪些层。

`super().__init__()` 是调用父类 `nn.Module` 的初始化逻辑。

这句不要省。你现在可以先机械记住：继承 `nn.Module` 时，`__init__` 里先写：

```python
super().__init__()
```

然后看：

```python
self.net = nn.Sequential(...)
```

`nn.Sequential` 是一种顺序容器。

它的意思是：把里面的层按顺序串起来。

输入先经过第一层，再经过第二层，再经过第三层。

现在里面是：

```python
nn.Linear(1, 64)
nn.ReLU()
nn.Linear(64, 64)
nn.ReLU()
nn.Linear(64, 1)
```

我们逐句讲。

---

## 第七部分：Linear 和 ReLU

先看：

```python
nn.Linear(1, 64)
```

`Linear` 是全连接层，也叫线性层。

它的作用是把输入特征从一个维度映射到另一个维度。

这里 `1` 是输入维度，`64` 是输出维度。

也就是说，每个样本原来只有 1 个数，经过这一层后，变成 64 个数。

如果输入 shape 是：

```python
[B, 1]
```

经过：

```python
nn.Linear(1, 64)
```

输出 shape 就是：

```python
[B, 64]
```

注意，batch 维 B 不变，变的是特征维度。

再看：

```python
nn.ReLU()
```

ReLU 是激活函数。

它的作用是给模型引入非线性。

如果只有 Linear，没有 ReLU，那么多层 Linear 叠在一起，本质上还是一个线性函数。

但是我们今天要拟合的是：

```text
y = sin(x) + 0.3x
```

这里有 `sin(x)`，它是非线性的。

所以模型必须有非线性能力。

ReLU 就是最常见的非线性激活函数之一。

你今天不用深究 ReLU 的所有性质，只要知道：

```text
Linear 负责线性变换。
ReLU 负责引入非线性。
MLP 通过 Linear + ReLU 拟合复杂函数。
```

接着：

```python
nn.Linear(64, 64)
```

表示从 64 维隐藏特征再映射到 64 维。

然后再一次：

```python
nn.ReLU()
```

最后：

```python
nn.Linear(64, 1)
```

把 64 维隐藏特征映射回 1 维输出。

因为我们预测的 y 只有一个数，所以最终输出维度是 1。

所以整个模型的 shape 流程是：

```text
[B, 1] -> [B, 64] -> [B, 64] -> [B, 1]
```

这里要背：

```text
Linear 改变最后一维特征维度，不改变 batch 维。
```

---

## 第八部分：forward 到底做什么

现在看：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)
```

`forward` 定义模型从输入到输出的计算过程。

也就是说，输入 `x` 进来之后，怎么经过模型里的层，最后得到 `pred`。

这里写：

```python
return self.net(x)
```

意思就是让 `x` 依次通过 `self.net` 里的所有层。

调用时我们写：

```python
pred = model(x)
```

注意，我们不是写：

```python
pred = model.forward(x)
```

虽然从直觉上看，`model(x)` 会调用 `forward(x)`，但 PyTorch 推荐写 `model(x)`。

因为 `model(x)` 会经过 `nn.Module` 的内部调用机制。以后你用 hook、混合精度、分布式训练时，这些机制都很重要。

所以今天你先背：

```text
forward 定义计算过程。
model(x) 会自动调用 forward(x)。
不要直接手动调用 model.forward(x)。
```

---

## 第九部分：loss 是什么

现在看：

```python
loss_fn = nn.MSELoss()
```

`loss_fn` 是 loss function，损失函数。

它的作用是衡量预测值和真实值之间的差距。

这里用的是：

```python
nn.MSELoss()
```

MSE 是 Mean Squared Error，均方误差。

公式你可以简单理解为：

```text
把 pred 和 y 的差值平方，再取平均。
```

如果 `pred` 和 `y` 很接近，MSE 就小。

如果差得很远，MSE 就大。

为什么这里用 MSELoss？

因为我们做的是回归任务。

回归任务常用 MSELoss。

代码：

```python
loss = loss_fn(pred, y)
```

意思是：

用预测值 `pred` 和真实值 `y` 计算误差。

你今天要记住：

```text
loss 是一个标量，表示模型当前错得有多严重。
训练的目标就是让 loss 下降。
```

---

## 第十部分：optimizer 是什么，AdamW 是什么

现在看：

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

这句对初学者很关键。

先看：

```python
model.parameters()
```

这表示取出模型中所有可训练参数。

比如每个 Linear 层都有 weight 和 bias。

这些就是模型要学习的东西。

然后：

```python
optim.AdamW(...)
```

这是创建一个 AdamW 优化器。

优化器的作用是：根据梯度更新模型参数。

你今天不需要推导 AdamW 的公式。

你只要掌握三个点：

第一，optimizer 管参数。

第二，`loss.backward()` 计算这些参数的梯度。

第三，`optimizer.step()` 根据梯度更新这些参数。

再看：

```python
lr=1e-3
```

`lr` 是 learning rate，学习率。

学习率控制每次参数更新的步子有多大。

学习率太大，训练可能震荡甚至发散。

学习率太小，训练会很慢。

今天 `1e-3` 就是 0.001，是一个常见起点。

再看：

```python
weight_decay=1e-4
```

`weight_decay` 可以粗略理解成一种正则化，防止参数变得过大。今天你不需要深入，只需要知道它是 optimizer 的一个常见参数。

背诵版：

```text
optimizer 是优化器。
AdamW 是一种常用优化器。
model.parameters() 把模型参数交给 optimizer 管理。
lr 控制每次更新步长。
optimizer.step() 真正更新参数。
```

---

## 第十一部分：训练循环完整讲解

现在我们看最核心的训练循环：

```python
loss_history = []
model.train()
num_steps = 2000

for step in range(num_steps):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if step % 100 == 0:
        print(f"step={step:04d}, loss={loss.item():.6f}")
```

从上往下讲。

```python
loss_history = []
```

用来记录每一步的 loss，后面画 loss 曲线。

```python
model.train()
```

这句话把模型切换到训练模式。

今天这个 MLP 里没有 Dropout 和 BatchNorm，所以 `train()` 的影响不明显。

但你必须养成习惯。

以后复杂模型里，训练模式和推理模式行为可能不同。

比如 Dropout 在训练时随机丢掉部分神经元，在推理时不会丢。

所以训练前写：

```python
model.train()
```

验证或推理前写：

```python
model.eval()
```

接着：

```python
for step in range(num_steps):
```

训练 2000 步。

每一步都做一次 forward、loss、backward、step。

看第一句：

```python
pred = model(x)
```

这叫前向传播，forward。

模型根据当前参数，对输入 `x` 做预测，得到 `pred`。

第二句：

```python
loss = loss_fn(pred, y)
```

计算预测值和真实值的误差。

第三句：

```python
optimizer.zero_grad()
```

清空梯度。

为什么要清空？

因为 PyTorch 默认会累积梯度。

如果你不清空，上一轮的梯度会和这一轮的梯度加在一起。

初学阶段，除非你明确在做梯度累积，否则每轮训练都要写：

```python
optimizer.zero_grad()
```

第四句：

```python
loss.backward()
```

反向传播，计算梯度。

此时模型每个参数的 `.grad` 会被填上。

第五句：

```python
optimizer.step()
```

根据刚刚算出来的梯度，更新模型参数。

完整顺序再背一遍：

```text
forward 预测。
loss 计算误差。
zero_grad 清空旧梯度。
backward 计算新梯度。
step 更新参数。
```

这里老师要强调一个常见错误：

有些人以为 `loss.backward()` 之后模型就学习了。

不对。

`backward()` 只是算梯度。

真正学习，也就是参数发生变化，是 `optimizer.step()`。

---

## 第十二部分：为什么要 loss.item()

代码里有：

```python
loss_history.append(loss.item())
```

`loss` 是一个 Tensor。

哪怕它只有一个数字，它也是 Tensor。

比如：

```python
tensor(0.1234, device='cuda:0')
```

如果我们要把它作为普通 Python 数字记录下来，就用：

```python
loss.item()
```

所以：

```python
loss.item()
```

就是把单元素 Tensor 转成普通数字。

今天背：

```text
loss.item() 用于打印和记录 loss 数值，不参与训练。
```

---

## 第十三部分：保存 loss 曲线和预测曲线

训练完之后，我们保存 loss 曲线：

```python
plt.figure()
plt.plot(loss_history)
plt.xlabel("step")
plt.ylabel("MSE loss")
plt.title("Training Loss")
plt.savefig(OUTPUT_DIR / "01_loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
```

这里你不需要重点学 matplotlib。只要知道它把 loss_history 画成图。

你要观察的是：

```text
loss 是否稳定下降。
```

如果 loss 不下降，要检查几个地方：

第一，数据 shape 对不对。

第二，模型输出 shape 对不对。

第三，loss 有没有正常计算。

第四，是否调用了 backward 和 step。

第五，学习率是否太大或太小。

接着保存预测曲线：

```python
model.eval()
with torch.no_grad():
    pred = model(x)
```

这里有两个重要点。

第一：

```python
model.eval()
```

表示切换到推理模式。

第二：

```python
with torch.no_grad():
```

表示这段计算不记录计算图。

为什么推理时不需要记录计算图？

因为我们只是看模型效果，不需要反向传播，不需要算梯度。

不记录计算图可以节省显存，也更快。

今天先背：

```text
训练用 model.train()。
推理/验证用 model.eval() + torch.no_grad()。
```

然后代码把 Tensor 从 GPU 转到 CPU：

```python
x_cpu = x.detach().cpu().squeeze().numpy()
```

这句也慢慢讲。

`detach()` 表示从计算图中分离出来。

`cpu()` 表示把 Tensor 放到 CPU。

`squeeze()` 表示去掉长度为 1 的维度，比如 `[1000, 1]` 变成 `[1000]`。

`numpy()` 表示转成 NumPy 数组，方便 matplotlib 画图。

你今天不需要把这几个函数研究很深，但要知道它们大概在干什么：

```text
detach：脱离计算图。
cpu：放到 CPU。
squeeze：去掉多余维度。
numpy：转成 NumPy 数组。
```

---

## 第十四部分：checkpoint 是什么

现在看保存 checkpoint：

```python
torch.save(
    {
        "step": num_steps,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss.item(),
    },
    CKPT_DIR / "01_mlp_regression.pt",
)
```

checkpoint 就是训练过程的存档。

为什么要保存 checkpoint？

因为真实大模型训练很贵，不可能每次都从头开始。

如果训练中断，可以从 checkpoint 恢复。

如果想部署模型，也要保存模型参数。

这里保存了几个东西：

```python
"step": num_steps
```

表示训练到了第几步。

```python
"model_state_dict": model.state_dict()
```

这是模型参数。

`state_dict` 是 PyTorch 保存参数的标准方式。

你可以理解成一个字典，里面存着每一层的 weight 和 bias。

```python
"optimizer_state_dict": optimizer.state_dict()
```

这是优化器状态。

为什么 optimizer 也要保存？

因为 AdamW 这类优化器内部不只是学习率，还有一些动量统计。如果你想完全恢复训练，就要保存 optimizer 状态。

```python
"loss": loss.item()
```

保存当前 loss，方便记录训练状态。

今天背：

```text
checkpoint 通常保存 model_state_dict、optimizer_state_dict、step 和 loss。
```

以后你看 OneVL 的 checkpoint，也会看到类似结构。

---

## 第十五部分：进入 Dataset 和 DataLoader

现在我们看第二个脚本。

第一个脚本里，我们直接把全部 1000 个样本一次性送进模型：

```python
pred = model(x)
```

这叫 full batch。

但真实训练中，数据集很大，不可能一次全送进去。

所以我们通常用 mini-batch。

这就需要两个东西：

```python
Dataset
DataLoader
```

老师这里要讲清楚它俩的分工。

`Dataset` 负责定义“一个样本怎么取”。

`DataLoader` 负责把很多个单样本组成一个 batch。

背诵版：

```text
Dataset 管单样本。
DataLoader 管 batch。
```

现在看代码：

```python
class SineDataset(Dataset):
    def __init__(self, n_samples: int = 1000):
        self.x = torch.linspace(-10, 10, n_samples).unsqueeze(1)
        self.y = torch.sin(self.x) + 0.3 * self.x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]
```

先看：

```python
class SineDataset(Dataset):
```

我们定义自己的数据集，继承 PyTorch 的 `Dataset`。

然后：

```python
def __init__(self, n_samples: int = 1000):
```

初始化数据集，生成 x 和 y。

接着：

```python
def __len__(self) -> int:
    return len(self.x)
```

`__len__` 返回数据集长度。

也就是有多少个样本。

这里就是 1000。

然后：

```python
def __getitem__(self, idx: int):
    return self.x[idx], self.y[idx]
```

`__getitem__` 表示给一个索引 idx，返回第 idx 个样本。

注意，它返回的是一个单样本。

如果：

```python
self.x.shape = [1000, 1]
```

那么：

```python
self.x[0].shape = [1]
```

所以 `dataset[0]` 返回的是：

```text
一个 x，shape 是 [1]
一个 y，shape 是 [1]
```

这里你一定要理解：

`Dataset` 不负责 batch。

`Dataset` 只负责单样本。

---

## 第十六部分：DataLoader 如何组成 batch

现在看：

```python
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

这里 `DataLoader` 接收一个 dataset，然后指定 batch size。

比如：

```python
batch_size = 32
```

表示每次从 Dataset 里取 32 个样本，堆成一个 batch。

然后训练时：

```python
for batch_idx, (x_batch, y_batch) in enumerate(loader):
```

每次循环拿到的不是单样本，而是 batch。

如果 batch_size 是 32，那么：

```python
x_batch.shape = [32, 1]
y_batch.shape = [32, 1]
```

这里的 32 是 batch 维。

所以再背一次：

```text
Dataset 返回 [1]。
DataLoader 返回 [B, 1]。
B 是 batch size。
```

`shuffle=True` 表示每个 epoch 打乱数据顺序。

为什么要打乱？

因为训练时不希望模型总是按固定顺序看到样本。打乱通常有助于训练更稳定。

今天你只要知道：

```text
shuffle=True 常用于训练集。
验证集一般不需要 shuffle。
```

---

## 第十七部分：batch size 对 loss 的影响

第二个脚本会尝试：

```python
for batch_size in [4, 32, 256]:
```

为什么要试不同 batch size？

因为 batch size 会影响训练曲线的波动。

如果 batch size 很小，比如 4，每次只看 4 个样本，梯度估计会比较噪声，loss 波动可能比较大。

如果 batch size 比较大，比如 256，每次看很多样本，梯度更稳定，loss 曲线可能更平滑。

但是 batch size 大也不一定永远更好。它会占更多显存，每个 epoch 的 step 数也会变少。

今天你不用深入 batch size 理论，只要观察现象：

```text
batch_size 小，loss 波动更明显。
batch_size 大，loss 相对更平滑。
```

这对后面训练大模型很重要。

以后你看到训练不稳定、loss 抖动、显存爆炸，batch size 都是你要首先检查的参数之一。

---

## 第十八部分：梯度流检查为什么重要

现在进入第三个脚本：梯度流检查。

这个脚本很关键，因为它直接对应以后看 OneVL 的分阶段训练。

在 VLA 或多模态模型里，经常会有这种训练方式：

第一阶段，只训练 projector。

第二阶段，冻结视觉 encoder，训练语言模型部分。

第三阶段，联合微调某些模块。

这些操作的本质就是控制梯度流。

也就是：哪些参数有梯度，哪些参数没有梯度；梯度能不能从后面传回前面。

所以今天我们要理解三个词：

```python
requires_grad
torch.no_grad()
detach()
```

先看模型：

```python
class SplitMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, detach_hidden: bool = False) -> torch.Tensor:
        hidden = self.encoder(x)

        if detach_hidden:
            hidden = hidden.detach()

        out = self.head(hidden)
        return out
```

这里我们故意把模型拆成两段：

```python
encoder
head
```

encoder 负责把输入 x 编成 hidden。

head 负责把 hidden 转成最终输出。

这很像大模型里的结构。

比如 VLA 里可能有：

```text
vision encoder -> projector -> language model / action head
```

今天我们用简单 MLP 模拟这个结构。

---

## 第十九部分：正常 backward 时，梯度怎么流

实验 A：

```python
pred = model(x, detach_hidden=False)
loss = loss_fn(pred, y)

optimizer.zero_grad()
loss.backward()
print_grad_norms(model, "after loss.backward()")
optimizer.step()
```

这里没有 no_grad，也没有 detach，也没有冻结参数。

所以 loss 会从输出开始，反向传回 head，再传回 encoder。

因此 encoder 和 head 的参数都有梯度。

打印梯度：

```python
for name, p in model.named_parameters():
    if p.grad is None:
        print(f"{name:20s} grad=None")
    else:
        print(f"{name:20s} grad_norm={p.grad.norm().item():.6f}")
```

这里：

```python
model.named_parameters()
```

会遍历模型里所有参数，并给出名字和参数本身。

比如：

```text
encoder.0.weight
encoder.0.bias
head.0.weight
head.0.bias
head.2.weight
head.2.bias
```

`p.grad` 是这个参数的梯度。

如果 `p.grad is None`，说明这个参数没有梯度。

如果有值，说明它参与了反向传播。

`p.grad.norm()` 是梯度的范数。你今天不需要深入范数，只要知道它可以粗略表示梯度大小。

背下来：

```text
看一个参数有没有被训练，就看它有没有 grad，以及 optimizer 是否管理它。
```

---

## 第二十部分：no_grad 是什么

实验 B：

```python
with torch.no_grad():
    pred = model(x, detach_hidden=False)

loss = loss_fn(pred, y)
optimizer.zero_grad()
loss.backward()
```

这里会报错，这是预期的。

为什么？

因为：

```python
with torch.no_grad():
```

告诉 PyTorch：这段计算不建立计算图。

没有计算图，就不能从 loss 反向传播回模型参数。

所以：

```python
loss.backward()
```

会失败。

`no_grad` 适合什么时候？

适合验证和推理。

比如训练完后，你只是想看看模型预测效果：

```python
model.eval()
with torch.no_grad():
    pred = model(x)
```

这样更省显存，也更快。

但是训练时不能把 forward 放进 no_grad，否则无法反向传播。

背诵版：

```text
no_grad 控制一整段计算不建图，常用于推理和验证，不用于训练反传。
```

---

## 第二十一部分：detach 是什么

实验 C：

```python
pred = model(x, detach_hidden=True)
loss = loss_fn(pred, y)

optimizer.zero_grad()
loss.backward()
print_grad_norms(model, "after loss.backward() with detach_hidden=True")
optimizer.step()
```

这里关键在模型内部：

```python
hidden = self.encoder(x)

if detach_hidden:
    hidden = hidden.detach()

out = self.head(hidden)
```

什么意思？

先让 x 经过 encoder，得到 hidden。

然后：

```python
hidden = hidden.detach()
```

这句话会切断 hidden 和 encoder 之间的梯度连接。

后面的 head 仍然可以训练。

但是梯度不会再传回 encoder。

所以实验 C 的结果应该是：

```text
encoder 没有梯度。
head 有梯度。
```

这就是 detach 的核心。

它不是让后面都不能训练。

它是从某个 Tensor 开始，把它前面的梯度截断。

背诵版：

```text
detach 是对某个 Tensor 截断梯度。
detach 后面的模块还可以训练。
detach 前面的模块收不到梯度。
```

这和 no_grad 不一样。

`no_grad` 是整段计算不建图。

`detach` 是某个中间结果切断和前面的连接。

---

## 第二十二部分：requires_grad=False 是什么

实验 D：

```python
for p in model.encoder.parameters():
    p.requires_grad = False
```

这表示冻结 encoder 的参数。

`requires_grad` 的意思是：这个 Tensor 是否需要计算梯度。

对于模型参数来说：

```python
requires_grad=True
```

表示这个参数参与训练。

```python
requires_grad=False
```

表示这个参数不参与训练。

然后优化器写成：

```python
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
```

这表示只把需要训练的参数交给 optimizer。

这里老师要提醒你：

冻结参数最好做两件事：

第一，把参数的 `requires_grad` 设为 False。

第二，optimizer 只接收可训练参数。

这样最清楚。

实验结果应该是：

```text
encoder grad=None。
head 有梯度。
```

背诵版：

```text
requires_grad=False 表示参数不参与梯度计算。
冻结模块时常用它。
```

---

## 第二十三部分：no_grad、detach、requires_grad 的区别

现在我们统一总结。

`torch.no_grad()`：

```text
控制一段代码不建立计算图。
常用于推理和验证。
```

`detach()`：

```text
控制某个 Tensor 不再把梯度传回前面的计算。
常用于截断梯度流。
```

`requires_grad=False`：

```text
控制某些参数不参与训练。
常用于冻结模块。
```

用一个类比：

`no_grad` 像是告诉 PyTorch：“这整段过程别记账。”

`detach` 像是告诉 PyTorch：“从这个中间节点往前别追了。”

`requires_grad=False` 像是告诉 PyTorch：“这个参数不用学。”

这三个东西以后非常重要。

你看 OneVL 的 Stage 0 / Stage 1 / Stage 2 时，不要只看名字，要看：

```text
哪些模块 requires_grad=False？
哪些输出被 detach？
哪些 forward 被 no_grad 包住？
optimizer 里到底有哪些 parameters？
```

这就能判断到底训练了哪些部分。

---

## 第二十四部分：今天的三个脚本分别训练你什么能力

第一个脚本 `01_mlp_regression.py` 训练的是完整闭环。

你要掌握：

```text
Tensor -> model -> loss -> backward -> step -> plot -> checkpoint
```

第二个脚本 `02_dataset_dataloader.py` 训练的是数据流。

你要掌握：

```text
Dataset 单样本。
DataLoader 组 batch。
batch 第一维是 B。
```

第三个脚本 `03_gradient_check.py` 训练的是梯度流。

你要掌握：

```text
参数有没有梯度。
梯度传到哪里。
哪些模块被冻结。
detach 和 no_grad 的区别。
```

这三个东西合起来，就是以后看任何训练代码的基础。

---

## 第二十五部分：今天必须背下来的代码骨架

现在请你跟着我背这个训练骨架：

```python
model.train()

for x_batch, y_batch in loader:
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)

    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

逐句解释：

```python
model.train()
```

进入训练模式。

```python
for x_batch, y_batch in loader:
```

从 DataLoader 里一批一批取数据。

```python
x_batch = x_batch.to(device)
y_batch = y_batch.to(device)
```

把输入和标签放到 CPU 或 GPU 上，必须和模型在同一个设备。

```python
pred = model(x_batch)
```

前向传播，得到预测值。

```python
loss = loss_fn(pred, y_batch)
```

计算预测值和真实值之间的误差。

```python
optimizer.zero_grad()
```

清空上一轮梯度。

```python
loss.backward()
```

反向传播，计算当前梯度。

```python
optimizer.step()
```

根据梯度更新参数。

这个骨架你必须能默写。

---

## 第二十六部分：你今天要达到什么程度才算吃透

今天不要求你能自己设计复杂神经网络。

不要求你理解 Transformer。

不要求你理解多模态大模型。

不要求你推导 AdamW。

今天吃透的标准是：

第一，你看到：

```python
x.shape = torch.Size([32, 1])
```

你能说出 32 是 batch size，1 是特征维度。

第二，你看到：

```python
class MLP(nn.Module)
```

你知道这是在定义 PyTorch 模型。

第三，你看到：

```python
def forward(self, x)
```

你知道这是定义输入到输出的计算路径。

第四，你看到：

```python
loss.backward()
```

你知道这是在计算梯度，不是在更新参数。

第五，你看到：

```python
optimizer.step()
```

你知道这才是在更新参数。

第六，你看到：

```python
Dataset
DataLoader
```

你知道 Dataset 管单样本，DataLoader 管 batch。

第七，你看到：

```python
with torch.no_grad()
```

你知道这是推理/验证时不建图。

第八，你看到：

```python
hidden.detach()
```

你知道这是切断 hidden 前面的梯度。

第九，你看到：

```python
p.requires_grad = False
```

你知道这是冻结参数。

第十，你看到：

```python
torch.save({"model_state_dict": model.state_dict()})
```

你知道这是保存模型参数。

达到这十条，今天就合格。

---

## 第二十七部分：和 OneVL 的关系

最后我们把今天内容和 OneVL 连接起来。

OneVL 看起来复杂，但训练代码仍然逃不开这些东西。

以后你打开 OneVL 的训练脚本，先不要慌。

你先找：

```python
Dataset
DataLoader
model
loss
optimizer
backward
step
checkpoint
```

然后问自己：

第一，batch 里有什么？

可能有图像、文本、轨迹、动作、mask、label。

第二，model forward 输入什么，输出什么？

可能输出 action token、trajectory、language logits 或 planning 结果。

第三，loss 怎么算？

可能不只是 MSELoss，可能有 cross entropy、trajectory loss、action loss、auxiliary loss。

第四，哪些参数参与训练？

看 `requires_grad`。

第五，梯度有没有被截断？

看 `detach()`。

第六，哪些地方只是推理？

看 `torch.no_grad()`。

第七，checkpoint 保存了什么？

看 `state_dict`。

所以今天你不是在学一个玩具 MLP。

你是在建立看复杂训练代码的坐标系。

---

## 第二十八部分：最后带你完整复述一遍

现在我们把今天内容从头复述一遍。

PyTorch 里，数据用 Tensor 表示。

Tensor 有 shape。shape 告诉我们每个维度的大小。

在今天的任务里，`x.shape = [1000, 1]`，表示 1000 个样本，每个样本 1 个特征。

我们用 `nn.Module` 定义模型。

`__init__` 里定义层。

`forward` 里定义数据怎么从输入变成输出。

`model(x)` 会自动调用 `forward(x)`。

模型输出 `pred`。

我们用 `MSELoss` 比较 `pred` 和真实值 `y`，得到 `loss`。

训练时，先清空旧梯度：

```python
optimizer.zero_grad()
```

再反向传播计算新梯度：

```python
loss.backward()
```

最后更新参数：

```python
optimizer.step()
```

如果数据很多，就写 Dataset 和 DataLoader。

Dataset 的 `__getitem__` 返回一个样本。

DataLoader 把多个样本拼成 batch。

所以 Dataset 返回 `[1]`，DataLoader 返回 `[B, 1]`。

推理或验证时，用：

```python
model.eval()
with torch.no_grad():
    pred = model(x)
```

这样不建计算图，不算梯度。

如果想切断某个中间 Tensor 前面的梯度，用：

```python
hidden.detach()
```

如果想冻结某些参数，用：

```python
p.requires_grad = False
```

如果想保存训练状态，用 checkpoint，保存模型参数、优化器状态、step 和 loss。

这就是今天的全部主线。

你今天不需要学更多。你需要把这条线跑通、背熟、讲清楚。

明天再进入更复杂的模型结构时，你就不会被代码吓住。
