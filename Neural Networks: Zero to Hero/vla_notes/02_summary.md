# 02 摘要｜makemore：语言模型和 bigram

## 一句话结论

这节课把语言模型的最小形式讲清楚：把离散符号变成 token，用前文预测下一个 token，并用负对数似然或交叉熵训练。

## 会议纪要

- 主题：从名字数据集出发，构建字符级语言模型 makemore。
- 核心问题：模型怎样“生成像训练集一样的东西”？答案是学习序列中下一个字符的条件概率分布。
- 关键内容：
  - 文本需要先变成离散 token id。
  - 使用特殊的开始/结束 token 表示序列边界。
  - bigram 模型只看前一个字符，预测下一个字符。
  - 可以用计数表估计概率，也可以用一个极简神经网络学习 logits。
  - 采样时按概率分布抽下一个 token，直到采到结束 token。
  - 训练目标是最大化训练数据的概率，等价于最小化 negative log likelihood。
  - one-hot、矩阵乘法、softmax、cross entropy 是神经网络版 bigram 的关键。

## 为什么要讲

所有自回归模型都继承了这个问题形式，包括文本 GPT，也包括把动作离散成 token 的 VLA 路线：

```text
给定前面的 token，预测下一个 token
```

区别只是 bigram 只看一个历史 token，而 GPT/VLA action-token 模型用 Transformer 看更长的 observation、instruction、history 和 previous actions。动作 token 化以后，连续控制也能被改写成语言模型式训练。

## 对完整 VLA 能力栈的价值

- Day2 的 causal LM 直接建立在这节课上。
- action-token VLA 的 `shift`、`cross_entropy`、自回归生成都来自这里。
- 自动驾驶里的 steering/throttle/waypoint/trajectory 如果被离散化成 action ids，本质就变成“根据场景和历史动作预测下一个动作 token”。
- 即使走连续 action head，理解 next-token prediction 也能帮你理解 language backbone 和 instruction following。

## 复习时必须回答

- 为什么要有开始/结束 token？
- bigram 的局限是什么？
- NLL 和 cross entropy 的关系是什么？
- 采样和训练时的概率分布分别怎么用？
- 连续 action 变成 token 后，为什么可以用语言模型训练？
