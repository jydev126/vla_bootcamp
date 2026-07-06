# 01 摘要｜micrograd：神经网络训练和反向传播

## 一句话结论

这节课的重点不是学一个玩具库，而是把深度学习训练最底层的机制拆开：loss 如何通过计算图反向传播到参数，参数又如何被梯度更新。

## 会议纪要

- 主题：从零实现一个小型 autograd engine，也就是 micrograd。
- 核心问题：神经网络为什么能训练？答案是把 forward 过程记录成计算图，再用链式法则从 loss 反向计算每个参数的梯度。
- 关键内容：
  - `Value` 对象同时保存数值、梯度、前驱节点和局部 backward 函数。
  - 加法、乘法、幂、`tanh` 等操作都可以看作计算图节点。
  - 反向传播不是玄学，是局部导数乘以上游梯度。
  - 同一个变量被多次使用时，梯度要累加。
  - 需要拓扑排序，保证先算后面的节点，再把梯度传回前面的节点。
  - 用这些标量运算可以拼出 neuron、layer、MLP，并用 gradient descent 优化参数。

## 为什么要讲

如果只会调用 `loss.backward()`，后面读 VLA 训练代码时会看不懂梯度到底穿过了哪些模块。比如 projector 冻不冻结、latent token 是否被 trajectory loss 训练、auxiliary decoder 是否影响 backbone，这些问题本质都在问：计算图连没连上，梯度有没有流过去。

## 对完整 VLA 能力栈的价值

- Day1 的 `loss.backward()` 和 `optimizer.step()` 是这节课的工程版本。
- 任意 VLA 训练都会有 perception encoder、language backbone、projector、action head、auxiliary head 等模块；它们能不能一起学，取决于计算图和梯度流。
- 你以后 debug “为什么 action head 学不动”“为什么视觉信息没有影响轨迹”“为什么语言指令没有被执行”时，第一反应应该是查梯度流、参数是否 requires_grad、是否被 detach。

## 复习时必须回答

- 为什么每个参数都需要 `grad`？
- 为什么同一个节点的梯度要累加？
- 为什么反向传播需要拓扑顺序？
- `loss.backward()` 到底在 PyTorch 内部做了什么？
- VLA 里的视觉编码器、语言模型、latent slot、action head 为什么能被同一个任务 loss 训练？
