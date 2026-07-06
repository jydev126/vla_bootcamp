# 10 完整笔记｜复现 GPT-2 124M：从教学模型到训练系统工程

## 0. 这节课到底要学会什么

前面的 GPT 是教学版。这节课把目标推进到：复现 GPT-2 124M。

重点不只是模型结构，而是完整训练工程：

```text
模型实现
权重加载
初始化
优化器
学习率调度
数据管线
混合精度
FlashAttention
torch.compile
梯度累积
DDP 分布式
评估 benchmark
吞吐优化
```

对 VLA 来说，这节极重要：真实 VLA 不会卡在“会不会写一个 Transformer block”，而会卡在训练吞吐、显存、多模态数据、评估闭环和工程稳定性。

## 1. GPT-2 124M 是什么

GPT-2 有多个尺寸，124M 是最小公开版本。

典型配置包括：

```text
n_layer = 12
n_head = 12
n_embd = 768
block_size = 1024
vocab_size ≈ 50257
```

模型仍然是 decoder-only Transformer。

## 2. GPTConfig

用 config 管理超参数：

```text
block_size
vocab_size
n_layer
n_head
n_embd
```

这样模型结构可复用，不把数字散落在代码里。

VLA 也应该这样管理：

```text
num_image_tokens
hidden_dim
num_latents
action_dim
trajectory_horizon
action_bins
loss_weights
```

## 3. GPT 模型结构

核心模块：

```text
wte: token embedding
wpe: position embedding
blocks: Transformer blocks
ln_f: final LayerNorm
lm_head: vocab projection
```

forward：

```text
idx -> token embedding + position embedding
-> blocks
-> final ln
-> lm_head
-> logits
```

## 4. CausalSelfAttention

标准 attention：

```text
qkv = c_attn(x)
q, k, v = split(qkv)
attention(q, k, v, causal=True)
out = c_proj(att)
```

GPT-2 把 Q/K/V 放在一个大 Linear 里一次算出来，提高效率。

## 5. MLP

GPT-2 MLP：

```text
Linear(n_embd, 4*n_embd)
GELU
Linear(4*n_embd, n_embd)
```

GELU 是 GPT 类模型常用激活。

## 6. 权重加载

复现第一步：从 HuggingFace GPT-2 加载权重，塞进自己写的模型。

要注意：

```text
参数名映射
某些权重需要 transpose
embedding/lm_head shape
LayerNorm 参数
```

加载后做 forward，对比输出是否合理。

这一步非常像你读 VLA repo 时加载 pretrained VLM/LLM/backbone。

## 7. 为什么要先对齐推理

如果模型结构和权重加载都没对齐，训练问题会被掩盖。

工程顺序：

```text
1. 模型能 forward
2. 权重能 load
3. 输出和参考实现接近
4. 再开始训练
```

VLA 里也一样：先确认 vision encoder、projector、LLM、head 的 shape 和输出正常，再跑大训练。

## 8. 初始化细节

GPT 类模型深，residual 多。

如果初始化不控制，残差分支会让方差随层数累积。

常见做法：对 residual projection 采用缩放初始化，例如按层数缩小。

这和第 4 节激活/梯度诊断直接相连。

## 9. weight tying

GPT 常把 token embedding 和 lm_head 权重共享：

```text
lm_head.weight = wte.weight
```

含义：输入 token embedding 和输出词表分类使用同一套 token 表示。

这样减少参数，也可能改善表示一致性。

## 10. AdamW

大模型常用 AdamW。

AdamW = Adam + decoupled weight decay。

优化器配置通常区分：

```text
需要 weight decay: Linear weight
不需要 weight decay: bias, LayerNorm weight, embedding
```

因为 bias/norm 参数不适合被同样惩罚。

## 11. 学习率调度

常见：

```text
warmup
-> cosine decay
-> min_lr
```

warmup 避免训练初期不稳定。

cosine decay 让后期逐渐收敛。

VLA 微调时也常需要 warmup，尤其是 projector/action head 刚接上 backbone 时。

## 12. gradient clipping

偶发梯度过大会破坏训练。

使用：

```text
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
```

它限制整体梯度范数。

多模态多 loss 训练中，gradient clipping 很常见。

## 13. TF32 / bfloat16 / mixed precision

训练吞吐来自硬件利用。

常见设置：

```text
TF32: 加速 Ampere 之后 GPU matmul
bfloat16: 降低显存和带宽，动态范围比 fp16 好
autocast: 自动混合精度
```

这些不改变模型目标，但会影响速度、显存和数值稳定。

VLA 图像 token 多，更需要混合精度。

## 14. torch.compile

`torch.compile` 可以让 PyTorch 捕获并优化计算图，减少 Python overhead，融合部分操作。

优点：吞吐提高。

注意：首次编译有开销，动态 shape 可能影响效果。

## 15. FlashAttention

普通 attention 显式构造：

```text
attention matrix: [B, n_head, T, T]
```

当 T 很大时，显存和带宽压力巨大。

FlashAttention 通过 IO-aware 算法避免完整 materialize attention matrix，分块计算 softmax 和加权和。

结果：

```text
更省显存
更快
支持更长上下文
```

VLA 如果有大量 visual tokens，也会受益于高效 attention。

## 16. DataLoader：从 toy 到真实数据

toy 版本直接从内存 tensor 切 batch。

真实 GPT 训练需要：

```text
下载数据
清洗
tokenize
保存 shards
训练时流式读取 token
```

数据管线会成为瓶颈。

VLA 数据更复杂：

```text
多相机图像
时间同步
CAN/ego state
地图
轨迹标签
文本指令
场景元数据
```

数据加载比纯文本更难。

## 17. token shards

把 token 保存成多个 shard：

```text
shard_000.npy
shard_001.npy
...
```

训练时按顺序或随机读取，构造 `[B, T]` batch。

优点：避免每次在线 tokenization，吞吐更稳定。

VLA 也应尽量预处理重计算成本高的数据，比如 image features、BEV cache、tokenized text。

## 18. gradient accumulation

如果目标 batch 很大，但显存放不下，就用多个 micro-batch 累积梯度。

伪代码：

```text
optimizer.zero_grad()
for micro_step in range(grad_accum_steps):
    loss = model(batch) / grad_accum_steps
    loss.backward()
optimizer.step()
```

为什么 loss 要除以 accum steps？

因为想让累积后的梯度等价于一个大 batch 的平均 loss 梯度。

## 19. DDP 分布式训练

DistributedDataParallel 中：

```text
每张 GPU 一份模型
每张 GPU 处理不同 micro-batch
backward 后 all-reduce 平均梯度
每张 GPU 执行相同 optimizer.step
```

关键概念：

```text
rank
world_size
local_rank
master process
```

VLA 多 GPU 训练必须理解这些。

## 20. DDP 和 gradient accumulation 的交互

如果每个 micro step 都同步梯度，会浪费通信。

通常只在最后一个 accumulation step 同步。

PyTorch DDP 可以用 `no_sync()` 控制。

大模型训练性能很依赖这些细节。

## 21. tokens/sec

训练不只看 loss，还要看吞吐：

```text
tokens per second
```

吞吐决定实验速度和成本。

VLA 中也可以看：

```text
samples/sec
frames/sec
image tokens/sec
closed-loop rollout speed
```

## 22. validation loss

定期在 val split 上评估 loss。

注意 eval 时：

```text
model.eval()
torch.no_grad()
```

避免 dropout、BN 训练行为影响评估。

## 23. HellaSwag 等 benchmark

课程会引入下游评估，例如多选题。

做法：对每个候选 completion 计算 token loss/概率，选概率最高的。

这说明 base LM 可以用“completion probability”做选择题。

VLA 也需要任务评估，不只看训练 loss。

## 24. 采样生成

训练过程中生成文本样例，直观看模型质量。

但样例不能替代指标。

VLA 中可视化 predicted trajectory 很有用，但也不能替代 closed-loop 指标。

## 25. 复现 GPT-2 和训练新模型的差别

复现包括：

```text
结构对齐
权重加载
benchmark 对齐
训练 recipe 接近
```

训练新模型还要考虑数据、算力预算、超参搜索。

## 26. scaling law 视角

模型大小、数据量、计算量共同决定性能。

不是无限增大模型就好；数据质量和 token 数也重要。

VLA 同理：

```text
更大 backbone 不一定解决动作质量
驾驶数据和闭环反馈可能更关键
```

## 27. 对完整 VLA 能力栈的意义

真实 VLA 训练需要：

```text
多模态 dataset pipeline
pretrained backbone load
projector/action head init
mixed precision
gradient accumulation
DDP/FSDP
loss logging
validation metrics
closed-loop simulation
failure mining
ablation management
checkpointing
```

你要从“写一个模型”升级到“运营一个训练系统”。

## 28. VLA 工程检查清单

读或写 VLA repo 时找：

```text
model config 在哪里
数据字段有哪些
图像预处理在哪里
text/action tokenizer 在哪里
pretrained 权重怎么 load
哪些模块 freeze
optimizer 参数组怎么分
lr schedule 怎么设
loss 如何组合
DDP/mixed precision 怎么配置
eval 指标是什么
失败样本怎么保存
checkpoint 怎么恢复
```

## 29. 常见误区

### 29.1 只实现 forward 就以为完成

训练系统远不止 forward。

### 29.2 忽略数据管线

GPU 等数据会浪费大量算力。

### 29.3 不看吞吐

低吞吐会让实验迭代不可承受。

### 29.4 只看 open-loop loss

自动驾驶最终要闭环表现。

### 29.5 不做 ablation

没有 ablation，你不知道提升来自哪里。

## 30. 复习自测

1. GPT-2 124M 的主要结构超参是什么？
2. 为什么要用 config 管理模型？
3. 权重加载时为什么可能需要 transpose？
4. 为什么要先对齐推理再训练？
5. AdamW 的 decay/no-decay 参数如何区分？
6. warmup 和 cosine decay 各自解决什么？
7. gradient clipping 解决什么问题？
8. FlashAttention 为什么更快更省显存？
9. gradient accumulation 为什么要缩放 loss？
10. DDP 同步的是什么？
11. tokens/sec 为什么重要？
12. VLA 训练系统比纯文本 GPT 多哪些数据/评估复杂度？
13. 读 VLA repo 时，除了 model forward 还必须找哪些文件？
