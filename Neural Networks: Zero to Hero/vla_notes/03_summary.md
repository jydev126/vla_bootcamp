# 03 摘要｜makemore MLP：embedding、上下文窗口和小神经语言模型

## 一句话结论

这节课从 bigram 升级到 MLP 语言模型：用多个历史 token 的 embedding 预测下一个 token，解决 bigram 看不见长上下文的问题。

## 会议纪要

- 主题：实现 Bengio 风格的神经概率语言模型。
- 核心问题：如果只看一个字符不够，怎样让模型利用更长上下文？
- 关键内容：
  - 用固定长度 context window，例如 3 个字符预测第 4 个。
  - 每个字符 id 通过 embedding table 变成向量。
  - 多个 embedding 拼接后输入 MLP。
  - MLP 输出 vocab 大小的 logits，再用 cross entropy 训练。
  - 构造 train/dev/test split，避免只看训练 loss。
  - 使用 mini-batch 随机训练，提高效率。
  - 学习率选择会显著影响 loss。
  - embedding 可以被可视化，表示模型学到的字符空间结构。

## 为什么要讲

GPT 里的 token embedding、position embedding、hidden state 都可以从这里开始理解。模型不直接处理字符本身，而是处理可训练的向量表示；上下文信息通过这些向量进入网络。

## 对完整 VLA 能力栈的价值

- VLM 里的 text token、visual token、latent token、action token 都会变成 embedding/hidden vectors。
- VLA 不是处理“字符串”或“图片块”本身，而是处理它们在 hidden space 里的向量。
- action head、risk head、occupancy head、trajectory head 接的是 hidden representation，不是原始 prompt。
- 学会 embedding 思想，才能理解多模态对齐：图像、语言、状态、地图、历史轨迹必须进入可融合的表示空间。

## 复习时必须回答

- embedding table 学到的是什么？
- 为什么 context window 比 bigram 强？
- 为什么要分 train/dev/test？
- mini-batch 梯度为什么有噪声但训练更快？
- 这节的 embedding 和后面 VLM visual token 有什么相似之处？
