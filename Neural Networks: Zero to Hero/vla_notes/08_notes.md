# 08 完整笔记｜State of GPT：预训练、对齐、助手化和应用边界

## 0. 这节课到底要学会什么

这节不是写代码，而是从系统层面理解 GPT assistant 是怎么来的。

核心流程：

```text
Pretraining
-> Supervised Fine-Tuning (SFT)
-> Reward Modeling
-> Reinforcement Learning / preference optimization
-> Assistant
```

对 VLA 来说，这节的价值是训练阶段思维：一个能看图、懂语言、会输出动作的系统，不是把模块拼起来就完事，而是通过不同阶段的数据和 loss 塑造出来的。

## 1. Base model 和 assistant model 不一样

base GPT 的训练目标：

```text
预测互联网文档中的下一个 token
```

它本质上是 document completer。

你给它：

```text
Question: ...
Answer:
```

它可能继续补全文档，但不一定以助手身份回答。

assistant model 则经过指令数据和偏好优化，学会：

```text
理解用户意图
按格式回答
拒绝不该做的事
更符合人类偏好
```

## 2. Pretraining

预训练是最大规模、最耗计算的阶段。

数据：

```text
网页
书籍
代码
论文
论坛
其他文本
```

目标：

```text
next-token prediction
```

结果：模型学到大量语言结构、世界知识、代码模式、推理痕迹。

但它的默认行为仍然是补全。

## 3. tokenization 在预训练中的位置

文本先经过 tokenizer：

```text
raw text -> token ids
```

模型只看到 token ids。

上下文窗口决定模型一次能看多少 token。

这对 VLA 的启发：任何模态进入大模型，都要先变成某种 token/embedding 序列；token 数直接影响成本和延迟。

## 4. next-token prediction 为什么强大

表面上只是预测下一个 token，但为了做好这个任务，模型必须学会：

```text
语法
事实
语义
格式
代码结构
推理模式
人类写作风格
```

因为这些都能帮助预测下一个 token。

但它也有局限：学到的是数据分布，不等于天然可靠、可控、对齐。

## 5. Supervised Fine-Tuning

SFT 数据是人工或模型辅助构造的指令-回答样本：

```text
User: 帮我解释...
Assistant: ...
```

训练目标仍然是 next-token prediction，但只在 assistant response 上计算 loss 或重点训练 response。

SFT 让模型学会“像助手一样说话”。

VLA 类比：

```text
VLM instruction tuning 让模型按视觉问答/驾驶指令格式输出
动作监督让模型按 observation-command 输出 action
```

## 6. Reward Modeling

SFT 只能模仿示范答案，但人类偏好更复杂。

Reward model 的训练数据通常是：

```text
同一个 prompt
多个 candidate answers
人类排序/偏好
```

Reward model 学习：

```text
哪个回答更好
```

它输出一个标量 reward。

## 7. RLHF / preference optimization

有了 reward model，可以优化语言模型，让它生成更高 reward 的回答。

这就是 RLHF 的核心。

现代也有 DPO 等替代方法，但核心仍是：

```text
用偏好数据改变模型输出分布
```

对 VLA 的启发：最终系统不只要模仿数据，还要符合安全、舒适、规则、任务成功等偏好。

## 8. 模型能力和行为不是一回事

预训练模型可能“有能力”完成任务，但不一定默认表现出来。

对齐阶段改变的是行为分布。

VLA 中也一样：

```text
backbone 可能有视觉/语言能力
但 action head 没训练好，车还是不会开
模型可能能描述危险
但没有 safety loss/闭环反馈，轨迹仍可能危险
```

## 9. prompt engineering 的位置

Prompt 可以引导模型行为，但不能替代训练。

Prompt 有用，因为模型是在 token 条件分布上工作。

但对于 VLA：

```text
prompt 可以表达 command/rule/context
训练决定模型是否真的把这些信息转成动作
```

不要把“会解释”误认为“会控制”。

## 10. 上下文窗口限制

GPT 只能在有限 token context 内工作。

如果上下文太长：

```text
成本上升
延迟上升
可能截断关键信息
attention 计算更贵
```

VLA 中图像 token 很多，历史帧很多，地图也很大，所以必须设计信息压缩：

```text
视觉压缩
BEV token 压缩
latent slots
memory tokens
检索/选择关键对象
```

## 11. LLM 不是一次性深思熟虑

模型每生成一个 token 做固定量计算。

复杂推理往往需要展开成多个 token 或调用工具。

VLA 中如果要求低延迟输出动作，不可能让模型生成很长 chain-of-thought 再控制。需要：

```text
短路径 hidden reasoning
latent reasoning
并行 action head
world model 辅助训练
```

## 12. 工具使用和外部系统

GPT assistant 可以结合工具：

```text
浏览器
代码解释器
检索
计算器
数据库
```

VLA 也不是纯模型孤岛，真实系统会结合：

```text
传感器同步
定位
地图
规则约束
控制器
仿真器
安全监控
```

完整能力栈要包括模型外系统。

## 13. Hallucination 和验证

语言模型可能生成看似合理但错误的内容。

自动驾驶里对应风险更高：

```text
错误识别障碍
错误理解信号灯
错误预测行人意图
轨迹看似平滑但不安全
```

所以 VLA 必须有评估、约束、闭环仿真和安全兜底。

## 14. 训练阶段思维迁移到 VLA

一个完整 VLA 可能有这些阶段：

```text
1. 视觉 encoder 预训练
2. 语言模型预训练
3. 视觉-语言 projector alignment
4. 视觉指令微调
5. 行为克隆 / trajectory imitation
6. action token 或 continuous action head 训练
7. world/risk/occupancy auxiliary training
8. closed-loop simulation fine-tuning
9. preference/safety/rule optimization
10. failure mining 数据回流
```

不同阶段解决不同问题。

## 15. 数据比模型名更重要

GPT 能力来自海量数据和训练 recipe。

VLA 也一样：

```text
传感器数据质量
标注轨迹质量
场景覆盖
长尾危险场景
驾驶风格一致性
闭环反馈数据
```

只看 architecture 不够。

## 16. 评估不能只看 loss

语言模型看：

```text
loss
benchmark
human preference
实际应用表现
```

VLA 要看：

```text
open-loop L2
collision rate
lane violation
red light violation
comfort
jerk
closed-loop success
latency
long-tail failure
```

这节课的系统视角提醒你：训练目标和真实目标之间有 gap。

## 17. 对完整 VLA 能力栈的意义

你要形成这样的判断：

```text
VLM 能描述场景，只解决 V 和 L 的一部分。
VLA 必须把理解转成 action。
Action 是否安全可靠，取决于数据、loss、评估、闭环和部署约束。
```

GPT assistant 不是 base model 自然长出来的；VLA driver 也不是 VLM 加一个 prompt 自然长出来的。

## 18. 复习自测

1. base GPT 和 assistant GPT 的区别是什么？
2. Pretraining 学到什么？没解决什么？
3. SFT 为什么仍然是 next-token prediction？
4. reward model 学的是什么？
5. RLHF 改变模型的什么？
6. 为什么能力和行为不是一回事？
7. Prompt 能解决什么，不能解决什么？
8. 上下文窗口对 VLA 有什么影响？
9. 为什么 VLA 不能只依赖长 CoT 生成？
10. 一个完整 VLA 训练 pipeline 可能有哪些阶段？
11. VLA 为什么必须看 closed-loop evaluation？
