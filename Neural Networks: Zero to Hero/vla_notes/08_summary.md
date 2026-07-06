# 08 摘要｜State of GPT：从 base model 到 assistant

## 一句话结论

这节课从系统层面解释 GPT assistant 的训练和使用：先大规模预训练成文档补全器，再经过 SFT、reward modeling、RLHF 等阶段变成更符合人类指令的助手。

## 会议纪要

- 主题：GPT 生态、训练流程和应用方式。
- 核心问题：为什么 base GPT 和 ChatGPT 不一样？
- 关键内容：
  - 预训练是最大头的计算：在大规模 token 上做 next-token prediction。
  - base model 本质是文档补全器，不天然是“助手”。
  - supervised fine-tuning 用人工示范数据把模型调成问答/指令格式。
  - reward model 学习人类偏好。
  - RLHF 用奖励信号进一步优化 assistant 行为。
  - 使用 LLM 时，要理解 token 限制、上下文窗口、模型不是一次性“思考完”的限制。
  - 工具使用、检索、代码执行等可以补足模型能力边界。

## 为什么要讲

你学 VLA 不能只盯模型结构，还要理解训练阶段和对齐目标。VLA 不是“随便把 VLM 接上一个 action head”就完事，而是通过感知预训练、语言对齐、动作监督、交互数据、辅助任务、评估反馈等阶段塑造模型行为。

## 对完整 VLA 能力栈的价值

- Day5 projector alignment 和 instruction tuning 对应“预训练表示到任务对齐”。
- VLA 的动作监督、语言解释、世界模型、风险预测都可以看作不同形式的对齐目标。
- 自动驾驶模型不仅要预测轨迹，还要在数据采集、闭环评估、失败分析、prompt/command 设计和安全约束中形成工程闭环。

## 复习时必须回答

- base model 和 assistant model 的区别是什么？
- SFT 和 RLHF 分别解决什么问题？
- 为什么 next-token prediction 能学到很多能力，但不等于会按指令工作？
- VLA 中的 alignment 分别对应视觉-语言、语言-动作、状态-动作、世界模型-规划哪些模块？
