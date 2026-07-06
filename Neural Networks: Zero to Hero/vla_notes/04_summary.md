# 04 摘要｜makemore Part 3：激活、梯度和 BatchNorm

## 一句话结论

这节课的重点是训练稳定性：模型结构写对还不够，初始化、激活分布、梯度分布和归一化会直接决定网络能不能学。

## 会议纪要

- 主题：诊断 MLP 训练中的激活和梯度问题。
- 核心问题：为什么网络有时 loss 不降、梯度很小或训练很慢？
- 关键内容：
  - logits 初始化过大，会导致模型一开始过度自信，初始 loss 异常高。
  - `tanh` 激活如果大量饱和到 -1 或 1，梯度会接近 0。
  - 权重初始化需要控制方差，避免激活和梯度逐层爆炸或消失。
  - Kaiming/He 初始化的思想是让信号在层间保持合适尺度。
  - BatchNorm 用 batch 统计量标准化 pre-activation，再用可学习的 scale/shift 恢复表达能力。
  - 训练和推理时 BatchNorm 行为不同，推理需要 running mean/var。

## 为什么要讲

大型 VLM/VLA 训练失败时，问题未必在“模型不够先进”，也可能是 hidden scale、初始化、norm、学习率、冻结策略出了问题。理解这些诊断工具，才能看懂真实 repo 里的 LayerNorm、RMSNorm、初始化和训练 trick。

## 对完整 VLA 能力栈的价值

- projector 输出 visual tokens 时，尺度必须适配 LLM hidden space。
- action head 输出的物理量有尺度，trajectory loss、collision loss、language loss 的梯度尺度也不同。
- latent slot、action query、adapter、LoRA、projector 初始化太大或太小都会影响训练稳定性。
- 读任何 VLA repo 时都要关注 normalization、初始化、冻结策略、loss weight 和不同 head 的梯度尺度。

## 复习时必须回答

- 为什么 logits 初始化太大不好？
- `tanh` 饱和为什么会杀死梯度？
- BatchNorm 在训练和推理时有什么不同？
- 为什么 projector alignment 本质上也关心 hidden space 的尺度？
