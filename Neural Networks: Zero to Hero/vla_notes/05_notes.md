# 05 完整笔记｜Backprop Ninja：手写张量级反向传播和梯度调试能力

## 0. 这节课到底要学会什么

前面 micrograd 是标量级别的 backprop。这节课升级到 Tensor 级别：不用 PyTorch autograd，手写 MLP 里每个中间张量的 backward。

目标不是以后都手写 backward，而是获得一种能力：

```text
看到 forward 里的任何一步，都知道梯度会怎么传、shape 应该是什么、哪里可能出错。
```

这对 VLA 很重要，因为 VLA 经常有多分支、多 loss、多模态共享 backbone。你必须知道某个 loss 到底会不会训练某个模块。

## 1. 为什么说 backprop 是 leaky abstraction

`loss.backward()` 很方便，但抽象会漏水。

当训练正常时，你可以不关心细节；当训练坏掉时，你必须知道：

```text
梯度是不是 None
梯度是不是 0
梯度是不是爆炸
梯度 shape 对不对
某个张量是不是被 detach
某个分支是不是没有连到 loss
```

手写 backward 是为了训练这种直觉。

## 2. forward 链路回顾

makemore MLP 大致 forward：

```text
Xb -> emb = C[Xb]
embcat = emb.view(...)
hprebn = embcat @ W1 + b1
hpreact = batchnorm(hprebn)
h = tanh(hpreact)
logits = h @ W2 + b2
loss = cross_entropy(logits, Yb)
```

手写 backward 要从 loss 反推回：

```text
logits -> h -> hpreact -> batchnorm -> W1/embcat -> emb -> C
```

## 3. 手写 backward 的基本规则

对每个中间变量保存一个梯度：

```text
dlogits = dloss/dlogits
dh = dloss/dh
dW2 = dloss/dW2
...
```

每一步都要检查：

```text
梯度 shape 必须等于对应变量 shape
```

这是最强 sanity check。

## 4. cross entropy 的梯度

交叉熵可以拆成：

```text
logits
-> softmax
-> probabilities
-> 取真实类别概率
-> -log
-> mean
```

最终对 logits 的梯度有一个非常简洁的形式：

```text
dlogits = probs
dlogits[range(B), Y] -= 1
dlogits /= B
```

直觉：

```text
错误类别：梯度为预测概率，推动它们变低
真实类别：梯度为预测概率 - 1，推动它变高
```

这是分类模型最重要的梯度直觉。

## 5. 为什么除以 batch size

loss 通常是 mean：

```text
loss = sum(sample_losses) / B
```

所以每个样本贡献的梯度也要除以 `B`。

这和 gradient accumulation 也有关：如果你把多个 micro-batch 的 loss 累积，要注意总梯度尺度等价于目标 batch。

## 6. Linear layer 的 backward

forward：

```text
out = x @ W + b
```

shape：

```text
x:    [B, Din]
W:    [Din, Dout]
b:    [Dout]
out:  [B, Dout]
```

backward：

```text
dx = dout @ W.T
dW = x.T @ dout
db = dout.sum(dim=0)
```

这三个公式必须会。

## 7. bias 的梯度为什么是 sum

bias `b` 被广播到 batch 的每个样本：

```text
out[i, :] = x[i, :] @ W + b
```

同一个 `b` 被用了 B 次，所以梯度要沿 batch 维累加：

```text
db = dout.sum(0)
```

这是广播 backward 的通用规则：forward 广播出去的维度，backward 要 sum 回来。

## 8. tanh 的 backward

forward：

```text
h = tanh(hpreact)
```

backward：

```text
dhpreact = (1 - h**2) * dh
```

注意用 forward 结果 `h` 就可以算导数。

如果 `h` 接近 ±1，`1 - h**2` 接近 0，梯度就小。这呼应第 4 节。

## 9. embedding lookup 的 backward

forward：

```text
emb = C[Xb]
```

`C` 是 embedding table：

```text
C.shape = [V, E]
Xb.shape = [B, T]
emb.shape = [B, T, E]
```

backward 时：

```text
dC = zeros_like(C)
对每个位置 (b, t):
    token_id = Xb[b, t]
    dC[token_id] += demb[b, t]
```

因为同一个 token 可能出现多次，所以必须累加。

PyTorch 里可以用 `index_add` 或类似机制实现。

## 10. reshape/view 的 backward

forward：

```text
embcat = emb.view(B, T*E)
```

backward 只是把梯度 reshape 回去：

```text
demb = dembcat.view(B, T, E)
```

reshape 不改变数值，只改变视图，所以 backward 也只是 reshape。

## 11. BatchNorm backward 为什么复杂

BatchNorm forward：

```text
mean = x.mean(0)
var = ((x - mean)**2).mean(0)
std_inv = (var + eps)**-0.5
xhat = (x - mean) * std_inv
out = gamma * xhat + beta
```

每个样本的输出依赖整个 batch 的 mean/var，所以梯度会跨样本传播。

这比 Linear/Tanh 复杂，因为 batch 内样本不是独立的。

## 12. BatchNorm 参数梯度

对：

```text
out = gamma * xhat + beta
```

有：

```text
dgamma = (dout * xhat).sum(0)
dbeta = dout.sum(0)
dxhat = dout * gamma
```

然后继续通过 normalization 的 mean/var 链路传回 `x`。

## 13. 手写 BN backward 的意义

不要求你以后背完整 BN backward，但要知道：

```text
normalization 会改变梯度路径
batch 内样本会相互影响
mean/var 也在计算图里
```

在 VLA 多 GPU 训练时，BatchNorm/SyncBatchNorm、small batch、eval mode 都可能成为问题。

## 14. 用 PyTorch autograd 校验

课程会把手写梯度和 PyTorch 自动梯度比较。

常见检查：

```text
cmp('logits', dlogits, logits)
cmp('h', dh, h)
cmp('W2', dW2, W2)
```

判断：

```text
exact: 是否完全一样
approx: 是否数值接近
maxdiff: 最大误差
```

这是写自定义层、CUDA kernel、复杂 loss 时非常重要的习惯。

## 15. 梯度 shape 总表

如果变量是：

```text
logits: [B, V]
h:      [B, H]
W2:     [H, V]
b2:     [V]
embcat: [B, T*E]
W1:     [T*E, H]
C:      [V, E]
```

那么对应梯度必须完全相同：

```text
dlogits: [B, V]
dh:      [B, H]
dW2:     [H, V]
db2:     [V]
dembcat: [B, T*E]
dW1:     [T*E, H]
dC:      [V, E]
```

如果 shape 不对，推导一定有错。

## 16. 多分支梯度如何合并

如果一个变量参与多个分支：

```text
y1 = f(x)
y2 = g(x)
loss = loss1(y1) + loss2(y2)
```

那么：

```text
dloss/dx = dloss1/dx + dloss2/dx
```

也就是梯度相加。

这对 VLA 多任务训练是核心：

```text
action loss
language loss
world loss
safety loss
```

如果共享 backbone，这些 loss 的梯度会在共享部分相加。

## 17. loss weight 的本质

如果：

```text
loss = action_loss + 0.1 * language_loss + 2.0 * safety_loss
```

那么对应梯度也被缩放：

```text
grad = grad_action + 0.1 * grad_language + 2.0 * grad_safety
```

所以 loss weight 不只是日志上的数字，而是在决定训练信号的优先级。

## 18. detach 的影响

如果你写：

```text
features = vision_encoder(image).detach()
action = head(features)
loss = action_loss(action, target)
```

那么 action loss 不会训练 vision encoder，只会训练 head。

这可能是故意冻结，也可能是 bug。

手写 backward 的直觉能让你快速判断这种图断在哪里。

## 19. 对 VLA 全栈的意义

VLA 可能有这样的 forward：

```text
image -> vision encoder -> visual tokens -> projector
text -> text embeddings
state/history -> state encoder
all tokens -> fusion transformer
latent/action query hidden -> heads
```

loss：

```text
trajectory_loss
control_loss
language_loss
world_model_loss
collision_loss
```

你要能判断：

```text
哪个 loss 训练哪个 head？
哪个 loss 回到 shared transformer？
哪个 loss 回到 vision encoder？
哪个模块被冻结？
哪个张量被 detach？
```

这节课就是为了让你能做这种判断。

## 20. 常见 debug 清单

训练 VLA 或任何深度模型时，可以检查：

```text
for name, p in model.named_parameters():
    print(name, p.requires_grad, p.grad is None, grad_norm)
```

重点看：

```text
应该训练的参数 grad 是否 None
不该训练的参数是否被冻结
grad norm 是否异常大/小
embedding/action token 是否收到梯度
projector 是否收到梯度
head 是否收到梯度
```

## 21. 复习自测

1. cross entropy 对 logits 的梯度为什么是 `probs - one_hot`？
2. Linear layer 的 `dx/dW/db` 分别怎么算？
3. bias backward 为什么要 sum？
4. reshape 的 backward 是什么？
5. embedding lookup 的 backward 为什么需要累加？
6. BatchNorm backward 为什么比 Linear 复杂？
7. 多个 loss 共享 backbone 时梯度如何合并？
8. loss weight 如何影响梯度？
9. `detach()` 会造成什么训练后果？
10. VLA 里如果 action loss 不训练 vision encoder，可能有哪些原因？
