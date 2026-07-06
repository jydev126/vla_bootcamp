# 10 摘要｜复现 GPT-2 124M：规模化训练工程

## 一句话结论

这节课从教学版 GPT 走向真实 GPT-2 复现，重点是模型结构、权重加载、优化器、数据管线、混合精度、FlashAttention、梯度累积和分布式训练。

## 会议纪要

- 主题：复现 GPT-2 124M，并逐步优化训练吞吐和效果。
- 核心问题：从 mini GPT 到真实可训练 GPT，中间差了哪些工程细节？
- 关键内容：
  - 实现 GPT-2 风格 block、config、embedding、attention、MLP、LayerNorm。
  - 从 HuggingFace 或官方权重加载 GPT-2 参数并校验输出。
  - 使用 AdamW，并正确区分需要 weight decay 和不需要 decay 的参数。
  - 学习率 warmup、cosine decay、梯度裁剪影响训练稳定性。
  - 使用 TF32/bfloat16、`torch.compile`、FlashAttention 提升速度。
  - 梯度累积模拟更大 batch。
  - DDP 在多 GPU 上同步/平均梯度。
  - 数据需要预 tokenize、shard、持续流式读取。
  - 用验证 loss、HellaSwag 等指标做评估。

## 为什么要讲

VLA 不是只写一个模型类。真实自动驾驶大模型训练会被吞吐、显存、batch、数据管线、评估、分布式工程、闭环仿真和安全指标限制。理解 GPT-2 复现，可以帮助你建立训练系统视角。

## 对完整 VLA 能力栈的价值

- VLA 训练也需要处理大 batch、梯度累积、冻结/解冻、loss 权重、多 GPU 和多模态数据加载。
- FlashAttention、mixed precision、DDP 都是大模型训练常规工具。
- ablation 不只看 loss，还要看任务指标、延迟、失败样本。
- 上车或仿真闭环时，还要看实时性、稳定性、长尾场景和安全约束。

## 复习时必须回答

- mini GPT 和 GPT-2 训练工程差在哪里？
- gradient accumulation 为什么要缩放 loss？
- DDP 同步的是什么？
- 为什么 AdamW 要区分 decay/no-decay 参数？
- 对 VLA 来说，吞吐和评估为什么同样重要？
