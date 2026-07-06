
# Day 1 PyTorch / DL 最小闭环总结

## 1. Tensor shape 怎么看？

Tensor shape 表示张量每个维度的大小。

例如：

```python
x.shape = torch.Size([1000, 1])


含义是：

* `1000`：样本数量。
* `1`：每个样本的特征维度。

如果经过 DataLoader 后：

```python
x_batch.shape = torch.Size([32, 1])
```

含义是：

* `32`：batch size，也就是这次一起送进模型的样本数。
* `1`：每个样本仍然只有一个输入特征。

看 shape 的核心口诀：

```text
第一维通常是 batch/sample 维，后面的维度通常是特征、通道、高宽或序列长度。
```

## 2. nn.Module 的 forward 做什么？

`nn.Module` 是 PyTorch 里所有模型模块的基类。

自己写模型时，通常要做两件事：

```python
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(...)

    def forward(self, x):
        return self.net(x)
```

* `__init__`：定义模型里有哪些层，哪些参数需要训练。
* `forward`：定义输入数据怎么经过这些层变成输出。

调用时写：

```python
pred = model(x)
```

它会自动执行 `forward(x)`。

不要直接写：

```python
pred = model.forward(x)
```

因为 `model(x)` 会经过 PyTorch 的 Module 调用机制，更符合训练、推理和 hook 的使用习惯。

## 3. loss.backward() 做什么？

`loss.backward()` 的作用是反向传播。

它会从 loss 出发，沿着计算图反向计算每个可训练参数对 loss 的梯度，并把梯度存到参数的 `.grad` 里。

典型训练顺序是：

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

含义是：

1. `optimizer.zero_grad()`：清空上一轮残留梯度。
2. `loss.backward()`：计算当前 loss 对每个参数的梯度。
3. `optimizer.step()`：根据梯度更新参数。

背诵版：

```text
loss.backward() 不更新参数，只计算梯度；optimizer.step() 才更新参数。
```

## 4. detach 和 no_grad 有什么区别？

`torch.no_grad()` 是一个上下文管理器，用来包住一段代码。

```python
with torch.no_grad():
    pred = model(x)
```

意思是：这段计算不记录计算图，所以不能用于训练反传。它常用于验证和推理，可以省显存、加快速度。

`detach()` 是对某一个 Tensor 做截断。

```python
hidden = hidden.detach()
```

意思是：从这个 hidden 开始，后面的计算还可以继续，但梯度不会再传回 hidden 前面的部分。

区别：

* `no_grad`：控制一整段计算，不建图。
* `detach`：控制某个中间 Tensor，切断它之前的梯度。

和 OneVL / VLA 的关系：

* 冻结视觉编码器时，常见做法是让视觉部分不参与训练。
* 只训练 projector 或 head 时，前面模块可能 `requires_grad=False` 或输出被 `detach()`。
* 分阶段训练时，必须知道梯度到底传到了哪里。

## 5. Dataset 和 DataLoader 怎么分工？

`Dataset` 负责定义数据怎么取。

核心方法：

```python
class SineDataset(Dataset):
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
```

`__getitem__` 每次只返回一个样本。

例如：

```python
x, y = dataset[0]
```

可能得到：

```python
x.shape = torch.Size([1])
y.shape = torch.Size([1])
```

`DataLoader` 负责把多个单样本组成 batch。

例如：

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for x_batch, y_batch in loader:
    ...
```

得到：

```python
x_batch.shape = torch.Size([32, 1])
y_batch.shape = torch.Size([32, 1])
```

背诵版：

```text
Dataset 管单样本，DataLoader 管 batch。Dataset 决定一个样本长什么样，DataLoader 决定一次喂多少个样本。
```

---

# 六、今天必须背下来的 20 句话

你今天就背这些。别贪多。

## 1. Tensor

```text
Tensor 是 PyTorch 里的多维数组，用来表示输入、输出、标签和模型参数。

shape 表示 Tensor 每个维度的大小。

[B, 1] 里 B 是 batch size，1 是每个样本的特征维度。

unsqueeze(1) 是在第 1 维增加一个维度，把 [N] 变成 [N, 1]。
```

---

## 2. nn.Module

```text
nn.Module 是 PyTorch 模型的基类。

__init__ 里定义层，forward 里定义计算过程。

model(x) 会自动调用 forward(x)。

nn.Linear(in_features, out_features) 表示一个全连接层，把输入维度映射到输出维度。
```

---

## 3. Loss

```text
loss 是模型预测值和真实标签之间的差距。

MSELoss 计算预测值和真实值的均方误差，常用于回归任务。

loss.item() 把单元素 Tensor 转成 Python 数字，方便打印和记录。
```

---

## 4. Backward / Optimizer

```text
optimizer.zero_grad() 清空上一轮梯度。

loss.backward() 计算当前 loss 对所有可训练参数的梯度。

optimizer.step() 根据梯度更新模型参数。

训练循环的核心顺序是：forward -> loss -> zero_grad -> backward -> step。
```

---

## 5. Dataset / DataLoader

```text
Dataset 负责单个样本怎么取。

DataLoader 负责把多个样本组成 batch。

__len__ 返回数据集大小。

__getitem__ 返回一个样本。
```

---

## 6. no_grad / detach / requires_grad

```text
torch.no_grad() 表示这一段计算不记录计算图，通常用于验证和推理。

detach() 表示从某个 Tensor 处切断梯度，不让梯度继续往前传。

requires_grad=False 表示这个参数不参与训练。
```

---

# 七、你今天真正要形成的肌肉记忆

以后你看 OneVL 的 `train.py`，先不要看复杂结构。先找这条线：

```python
for batch in dataloader:
    output = model(batch)
    loss = compute_loss(output, batch)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

然后再问：

```text
batch 从哪里来？          Dataset / DataLoader
model 是什么？            nn.Module
forward 做了什么？        输入怎么变成输出
loss 怎么算？             预测和标签怎么比较
梯度传到哪里？            backward / detach / requires_grad
哪些参数更新？            optimizer 管了哪些 parameters
怎么保存？                checkpoint / state_dict
```

你今天的合格标准不是“我看完教程了”，而是：

```text
我能手写一个训练闭环。
我能解释每个 Tensor 的 shape。
我能解释 backward 和 step 的区别。
我能解释 Dataset 和 DataLoader 的分工。
我能判断一段模型有没有梯度。
```

这才是后面学 OneVL 的地基。
