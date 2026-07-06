# 06 摘要｜WaveNet：更深的序列模型和层级信息融合

## 一句话结论

这节课把 MLP 语言模型升级成更深、更模块化的 WaveNet-like 架构，让上下文信息逐步融合，而不是一次性压扁进一个 hidden layer。

## 会议纪要

- 主题：在 makemore 上构建类似 WaveNet 的字符级语言模型。
- 核心问题：如何让模型利用更长上下文，同时保持结构可训练、可扩展？
- 关键内容：
  - 旧 MLP 把多个 token embedding 直接 flatten，信息融合太快。
  - WaveNet 思路是逐层、局部、层级式融合序列信息。
  - 用模块化方式重写网络组件，例如 Linear、BatchNorm、Tanh、Embedding、Flatten。
  - 重点理解 batch dimension 和 sequence/context dimension 的区别。
  - 多个 batch-like 维度可以并行处理，最后再 reshape。
  - 类似 1D convolution 的思想：相邻 token 先融合，再逐层扩大感受野。

## 为什么要讲

Transformer 之前，序列建模已经在探索“如何融合上下文”。这节课让你理解：模型结构的本质是决定信息如何在 token 间流动。VLM/VLA 中 visual token、text token、latent token 的融合也是同一个问题。

## 对完整 VLA 能力栈的价值

- trajectory/action chunk 是序列，不是单个标量。
- VLA 需要融合图像 token、状态 token、文本 token、latent token。
- action query 或 latent token 可以看作输出槽位，它们通过 attention/decoder 逐步吸收上下文信息。
- 规划不是孤立分类：它需要把局部视觉、历史轨迹、导航目标和动态障碍逐层整合成可执行动作。

## 复习时必须回答

- 为什么直接 flatten 长上下文不理想？
- 层级融合和一次性融合有什么区别？
- batch 维度和 time/token 维度为什么不能混？
- WaveNet 的 receptive field 思想和 Transformer attention 有什么关系？
