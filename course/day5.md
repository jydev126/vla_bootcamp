# Day 5 上课台词: VLM projector alignment 与 instruction tuning

## 开场

今天是 Day 5。

昨天 Day4，我们已经把 VLM 的结构接起来了。

主线是:

```text
image -> patches -> visual_features -> projector -> visual_tokens
prompt -> text_tokens
visual_tokens + text_tokens -> tiny LLM hidden
ANSWER hidden -> answer_head -> reason logits
```

昨天解决的是结构问题。

今天解决训练问题。

你以后读 MiniGPT-4、LLaVA、OpenVLA、OneVL 这类项目，经常会看到这些词:

```text
freeze vision encoder
train projector
alignment
instruction tuning
stage 1
stage 2
requires_grad_(False)
```

如果你只把它们当工程细节，就会读不懂训练代码。

今天我们用两个 toy 脚本把它们讲清楚:

```bash
.venv/bin/python day5_vlm_alignment/01_projector_alignment.py
.venv/bin/python day5_vlm_alignment/02_instruction_tuning.py
```

今天你要掌握三件事。

第一，projector 不只是改 shape，它要学习对齐视觉空间和语义空间。

第二，freeze vision encoder 是为了降低训练难度，也模拟真实 VLM 里保护预训练视觉能力的做法。

第三，alignment 和 instruction tuning 是两个不同训练目标。

---

## 第一部分: 先回顾 projector 为什么存在

Day4 里 vision encoder 输出:

```text
visual_features: [B, 64, 64]
```

这里最后一维 64 是 `vision_dim`。

但是 tiny LLM 需要:

```text
llm_dim = 96
```

所以 projector 做:

```text
[B, 64, 64] -> [B, 64, 96]
```

如果你只看 shape，会觉得 projector 就是一个 Linear。

但这只是表面。

真正的问题是:

```text
随机 Linear 输出的 96 维向量，语言模型真的能理解吗?
```

答案通常是不行。

维度对上，不等于语义对上。

这就像你把中文句子硬塞进英文模型，字符数量对上没有意义。

projector 真正要学的是:

```text
把视觉 encoder 的输出，挪到语言/语义 hidden space 中合理的位置。
```

这就是 alignment。

---

## 第二部分: 01_projector_alignment.py 的目标

第一个脚本是:

```text
day5_vlm_alignment/01_projector_alignment.py
```

它模拟 MiniGPT-4 风格的 Stage 1。

真实 Stage 1 大致是:

```text
冻结 vision encoder
冻结 LLM
训练中间 bridge / projector
让视觉特征能进入语言模型空间
```

我们的 toy 版本没有真实 LLM 语义空间。

所以它用一个更小的替代:

```text
label_anchors
```

每个 reason 类别有一个可学习 anchor:

```text
keep
brake_due_to_close_lead
lane_change_left
lane_change_right
```

图像经过 frozen vision encoder 和 trainable projector 后，也得到一个 96 维 pooled visual 表示。

训练目标是:

```text
pooled visual 靠近正确类别 anchor
pooled visual 远离错误类别 anchor
```

所以这个脚本表面上是在分类。

但你要理解，分类只是训练手段。

真正教学目标是:

```text
训练 projector, 让视觉表示进入一个可分的语义空间。
```

---

## 第三部分: ProjectorAlignment 里有哪些模块

类名是:

```python
class ProjectorAlignment(nn.Module):
```

里面有四个重要模块。

第一个:

```python
self.vision_encoder = day4.bridge.TinyViTEncoder()
```

它来自 Day4，负责:

```text
image -> visual_features
```

第二个:

```python
self.projector = day4.bridge.VisionProjector()
```

它负责:

```text
visual_features [B, 64, 64] -> visual_tokens [B, 64, 96]
```

第三个:

```python
self.token_pooler = nn.Linear(96, 1)
```

为什么需要 pooler?

因为一张图有 64 个 visual tokens。

我们最后要和类别 anchor 比相似度，需要每张图一个整体向量:

```text
[B, 64, 96] -> [B, 96]
```

最简单可以 mean pooling。

但 toy 图里左转、右转提示只占很小区域，平均池化容易把它们冲淡。

所以这里让模型自己学每个 patch token 的权重。

第四个:

```python
self.label_anchors = nn.Parameter(torch.randn(len(REASONS), 96) * 0.02)
```

它是四个可学习类别原型。

你可以把它想成四个语义方向。

如果图像应该刹车，pooled visual 应该更靠近 `brake_due_to_close_lead` 的 anchor。

---

## 第四部分: 为什么冻结 vision encoder

代码里有:

```python
for p in self.vision_encoder.parameters():
    p.requires_grad_(False)
```

这句的意思是:

```text
vision encoder 不更新参数。
```

forward 里也用:

```python
with torch.no_grad():
    visual_features, patch_tokens, patches = self.vision_encoder(image)
```

为什么要这么做?

真实 VLM 里，vision encoder 往往是 CLIP ViT、EVA ViT 这种预训练模型。

它已经有视觉能力。

如果你的图文数据不大，一上来全量训练，很容易破坏原来的视觉表示。

而且 vision encoder 很大，训练成本高。

所以 Stage 1 常见做法是:

```text
视觉能力先相信它。
语言能力先相信它。
只训练中间翻译器。
```

在我们的 toy 代码里，vision encoder 虽然不是预训练大模型，但冻结它可以让你清楚看到:

```text
只训练 projector / pooler / anchors，也能改变语义对齐结果。
```

---

## 第五部分: alignment forward 怎么走

`ProjectorAlignment.forward` 主线是:

```python
with torch.no_grad():
    visual_features, patch_tokens, patches = self.vision_encoder(image)

visual_tokens = self.projector(visual_features)
weights = self.token_pooler(visual_tokens).softmax(dim=1)
pooled = (visual_tokens * weights).sum(dim=1)
anchors = F.normalize(self.label_anchors, dim=-1)
logits = F.normalize(pooled, dim=-1) @ anchors.t() * 10.0
```

shape 逐步看:

```text
image:           [B, 3, 64, 64]
visual_features: [B, 64, 64]
visual_tokens:   [B, 64, 96]
pool_weights:    [B, 64, 1]
pooled_visual:   [B, 96]
label_anchors:   [4, 96]
logits:          [B, 4]
```

`softmax(dim=1)` 的意思是:

```text
在 64 个 visual tokens 上分配注意力权重。
```

有些 patch 对答案更重要，比如左转箭头、右转箭头、前车位置。

然后 cosine similarity 给每个类别一个分数。

最后用 cross entropy 训练。

运行脚本时你会看到:

```text
parameters: total=..., trainable=...
trainable modules: projector, token_pooler, label_anchors
before training val_acc=...
epoch ... alignment_loss=... val_acc=...
```

这里最应该观察的是:

```text
vision_encoder=frozen, 但 val_acc 仍然会因为 projector/pooler/anchors 的训练而变化。
```

---

## 第六部分: alignment 和普通分类有什么区别

你可能会问:

这不还是分类吗?

从 toy 任务表面看，是。

但我们关心的不是分类器本身。

我们关心的是 hidden representation 怎么被训练目标塑形。

toy 分类任务迫使 projector 学到一种映射:

```text
视觉特征 -> 可区分语义类别的 96 维空间
```

真实 VLM 里，这个空间会接近 LLM hidden space。

我们这里用 label anchors 简化了语言空间。

所以请记住:

```text
toy 分类是训练手段。
alignment 是训练目的。
```

以后你看到 auxiliary loss、contrastive loss、caption loss，也要这样想。

loss 表面上可能是分类、重建、预测 token。

但真正目标往往是塑造中间表示。

---

## 第七部分: 02_instruction_tuning.py 的目标

第二个脚本是:

```text
day5_vlm_alignment/02_instruction_tuning.py
```

它模拟 Stage 2: instruction tuning。

和 Stage 1 的区别是，Stage 2 不只是让 image embedding 靠近 anchor。

Stage 2 要让模型在图像和 prompt 的条件下回答。

代码里 prompt 有三种:

```text
BOS what affects ego planning ANSWER
BOS choose driving reason now ANSWER
BOS look image and decide ANSWER
```

训练 dataset 使用:

```python
TinyVLMDataset(train_samples, prompt_mode="random")
```

也就是说，同一类任务会用不同问法出现。

模型需要学会:

```text
不管用户怎么问，都在 ANSWER token 位置形成正确 hidden state。
```

这就比 Stage 1 更接近真实 VLM 使用方式。

---

## 第八部分: instruction tuning 训练哪些模块

脚本里先创建 Day4 的 TinyVLM:

```python
model = day4.TinyVLM(len(day4.VOCAB)).to(device)
```

然后冻结 vision encoder:

```python
freeze_vision_encoder(model)
```

这个函数内部就是:

```python
for p in model.vision_encoder.parameters():
    p.requires_grad_(False)
```

然后训练的模块是:

```text
projector
tiny_llm
answer_head
```

脚本会打印:

```text
parts: vision_encoder=frozen, projector=train, tiny_llm=train, answer_head=train
```

为什么 Stage 2 要训练 tiny_llm?

因为现在问题不只是“图像属于哪类”。

现在有 prompt tokens。

模型要学会让 text tokens 和 visual tokens 交互。

尤其是 ANSWER 位置的 hidden state，需要聚合图像和指令。

所以 Stage 2 允许语言侧的小 backbone 一起适配任务。

真实大模型里，不一定全量训练 LLM。

可能只训 LoRA、adapter、Q-Former、projector。

但抽象一样:

```text
让模型学会按指令使用视觉信息。
```

---

## 第九部分: 看同一张图不同 prompt

`02_instruction_tuning.py` 里有一个函数:

```python
show_prompt_predictions(model, val_samples[0], device)
```

它会对同一张图跑三种 prompt。

训练前你可能看到三种 prompt 都乱猜。

训练后你希望看到:

```text
BOS what affects ego planning ANSWER       -> brake_due_to_close_lead
BOS choose driving reason now ANSWER       -> brake_due_to_close_lead
BOS look image and decide ANSWER           -> brake_due_to_close_lead
```

这就是 instruction tuning 的最小直觉。

模型不只是记住一串固定问题。

它在不同问法下，都把视觉信息读出来，放到 ANSWER hidden 里。

---

## 第十部分: 和后面 VLA 的关系

今天我们输出的是 reason。

也就是:

```text
hidden -> answer_head -> reason class
```

但 VLA 最后不是只回答 reason。

VLA 要输出动作。

结构可以变成:

```text
image + state/instruction -> hidden -> action_head -> trajectory/action
```

所以 Day5 的意义不是这个 toy 分类本身。

意义是你开始理解:

```text
多模态 hidden state 可以被不同 head 读取。
```

用 answer head，就是 VQA。

用 action head，就是 VLA planning。

用 auxiliary decoder，就是 OneVL 里的辅助训练目标。

---

## 收尾

今天你要记住三句话。

第一:

```text
projector 不只是改 shape，它是在对齐空间。
```

第二:

```text
freeze 不是偷懒，而是降低训练难度、保护已有表示。
```

第三:

```text
instruction tuning 让模型学会按不同 prompt 使用视觉信息。
```

从 Day4 到 Day5，你已经有了一个最小 VLM:

```text
image -> visual tokens
prompt -> text tokens
fusion hidden -> task head
```

明天 Day6，我们开始把输出从 reason 转向 action。
