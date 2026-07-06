# 07 摘要｜从零构建 GPT：Transformer 和 causal LM

## 一句话结论

这节课把前面的语言模型升级为真正的 mini GPT：token embedding 加 position embedding，经过 masked self-attention 和 Transformer block，用 next-token prediction 训练。

## 会议纪要

- 主题：从零实现一个字符级 GPT。
- 核心问题：ChatGPT 这类模型的底层训练形式是什么？答案仍然是 causal next-token prediction。
- 关键内容：
  - 文本 token 化后形成序列。
  - `get_batch` 构造输入 `x` 和右移一位的目标 `y`。
  - token embedding 表示“是什么 token”，position embedding 表示“在什么位置”。
  - self-attention 用 Q/K/V 让 token 从上下文读取信息。
  - causal mask 阻止当前位置看到未来 token。
  - multi-head attention 让模型从多个子空间读取关系。
  - residual、LayerNorm、FeedForward、Dropout 组成 Transformer block。
  - generation 是自回归循环：把新采样 token 接回上下文继续预测。

## 为什么要讲

这节是你理解 LLM/VLM/VLA 的主干。VLM 是把 image token 接进类似 Transformer 的 hidden space；action-token VLA 是把动作纳入 causal LM；continuous-head VLA 是取某些 hidden state 接 trajectory/control head；latent-slot VLA 是让特殊位置承载压缩推理和规划表示。

## 对完整 VLA 能力栈的价值

- Day2 mini GPT 基本对应这节。
- Day3 hidden state、prefill/decode 依赖这节。
- action token 的 shift 和 causal mask 与 GPT 一致。
- latent hidden、CLS hidden、最后 token hidden、action query hidden 的抽取，本质都是在 Transformer 输出中取特定位置或槽位。
- VLA 的延迟和部署也与自回归 decode、prefill、上下文长度紧密相关。

## 复习时必须回答

- 为什么 `x` 和 `y` 要错位一位？
- causal mask 解决什么问题？
- Q/K/V 分别承担什么角色？
- 为什么需要 position embedding？
- hidden state 为什么可以接不同 head？
