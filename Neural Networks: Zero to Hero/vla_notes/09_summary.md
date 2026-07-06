# 09 摘要｜GPT Tokenizer：BPE、特殊 token 和隐藏坑

## 一句话结论

这节课说明 tokenization 不是无聊预处理，而是 LLM 输入输出接口的一部分；很多模型能力、bug 和边界都来自 tokenizer。

## 会议纪要

- 主题：从零构建 GPT tokenizer，理解 BPE。
- 核心问题：文本如何变成模型能处理的整数序列？
- 关键内容：
  - 模型不能直接吃字符串，需要 token ids。
  - 字符级 tokenizer 简单但序列长；词级 tokenizer 泛化差。
  - BPE 从字节或字符开始，反复合并高频相邻 pair。
  - vocab 由基础符号和 merge 产生的新 token 组成。
  - encode 是把字符串变成 token ids，decode 是反向还原。
  - Unicode、byte fallback、regex pre-tokenization、special tokens 都会影响结果。
  - tokenization 会导致数字、空格、大小写、多语言等奇怪现象。

## 为什么要讲

VLA 里 token 不只指文本。image patch、visual token、latent token、action token 都是模型接口设计。OpenVLA 把连续 action 离散化，本质就是为动作设计 tokenizer。

## 对完整 VLA 能力栈的价值

- Day7 的 action tokenizer 直接对应这节的思想。
- Day3/Day9 的 latent token 需要理解 special token id 和 hidden position。
- VLM 里的 visual token 也是“把非文本模态接入 Transformer 序列”的接口。
- token 粒度会影响序列长度、精度、延迟和训练难度。
- 对智驾来说，action token 的 bin 设计、轨迹点频率、控制量范围和安全边界都会影响模型能否输出可执行动作。

## 复习时必须回答

- BPE 为什么比字符级 tokenizer 更高效？
- special token 为什么危险又必要？
- encode/decode 为什么不总是直觉一致？
- action tokenization 会损失什么信息？
