# 05 摘要｜Backprop Ninja：手写张量级反向传播

## 一句话结论

这节课把 `loss.backward()` 拆成张量级公式，让你真正知道 cross entropy、BatchNorm、embedding 和矩阵乘法的梯度怎么流。

## 会议纪要

- 主题：不用 PyTorch autograd，手动写出 MLP 训练中各个中间变量的反向传播。
- 核心问题：backprop 是不是可以完全推出来？答案是可以，而且推出来后更容易 debug。
- 关键内容：
  - 从 loss 开始，逐步推导 logits、softmax、cross entropy 的梯度。
  - 反向传播时每个张量的 shape 必须和 forward 对应。
  - 广播、求和、矩阵乘法、索引操作都有明确梯度规则。
  - BatchNorm 的 backward 比普通线性层复杂，因为均值和方差依赖整个 batch。
  - embedding lookup 的反向传播会把梯度累加回被索引到的 embedding 行。
  - 用 PyTorch autograd 的结果检查手写梯度。

## 为什么要讲

真实 VLA 训练经常不是一个 loss，而是 perception、language、world modeling、trajectory、control、safety 等多个 loss 同时作用在 shared backbone 和不同 head 上。只会看 forward 很不够，必须能判断某个 loss 会不会影响某个参数，以及梯度形状是否合理。

## 对完整 VLA 能力栈的价值

- auxiliary decoder、planning head、control head 的梯度如何回到 shared hidden，是理解 VLA 训练的关键。
- action token 的 cross entropy 和 continuous trajectory 的 MSE/L1 loss 梯度性质不同。
- embedding 行被重复使用时梯度会累加，这对应 special token、action token、latent token 的训练。
- 多任务 loss 权重会改变梯度合成方式，直接影响模型偏向“会说”“会看”还是“会开”。

## 复习时必须回答

- 为什么手写 backward 能帮助 debug？
- embedding lookup 的梯度为什么要累加到 table 的某些行？
- cross entropy 的梯度直觉是什么？
- 多个 loss 加起来时，梯度怎么合并？
