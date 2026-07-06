# Day 4 上课台词: 从 GPT token 到 image token

## 开场

今天是 Day 4。

今天我们开始进入 VLM，但是不要一上来就调用 Qwen-VL、LLaVA、GPT-4o 这种现成 API。

调用 API 可以很快得到一句描述，但你还是不知道模型内部到底发生了什么。

所以今天我们延续 Day2 mini GPT 的风格: 每段代码都简单，每段代码都能跑，每段代码都能看到一个明确作用。

今天只解决一个核心问题:

```text
图像怎么变成 LLM 能处理的一串 token?
```

你先把前几天的主线接回来。

Day1 是训练闭环:

```text
data -> model forward -> loss -> backward -> optimizer.step
```

Day2 是 mini GPT:

```text
text -> token ids -> token embedding + position embedding -> Transformer -> logits
```

Day3 是 HuggingFace LLM 和 hidden state:

```text
input_ids: [B, T]
hidden_states: [B, T, C]
logits: [B, T, vocab_size]
```

今天我们把图像接进这条链路。

VLM 不是 LLM 直接读取 JPEG 文件。LLM 最终接收的仍然是向量序列:

```text
[B, T, C]
```

所以图像必须先变成类似文本 embedding 的序列。

这串图像向量就叫 visual tokens。

今天的最小链路是:

```text
image
-> pixel_values
-> patches
-> patch_tokens
-> visual_features
-> projector
-> visual_tokens

text
-> input_ids
-> text_tokens

visual_tokens + text_tokens
-> tiny LLM hidden
-> ANSWER hidden
-> answer_head
-> reason logits
```

今天四个脚本按顺序跑:

```bash
.venv/bin/python day4_vlm/01_patchify_image.py
.venv/bin/python day4_vlm/02_tiny_vit_encoder.py --batch-size 2
.venv/bin/python day4_vlm/03_minigpt4_bridge.py --batch-size 2
.venv/bin/python day4_vlm/04_train_tiny_vlm.py
```

你不要把它们当成四个互不相关的 demo。

它们是一条流水线。

第一个脚本让图片变成 patch 序列。

第二个脚本让 patch 序列变成 visual features。

第三个脚本把 visual features 通过 projector 接到 tiny LLM。

第四个脚本把整个 tiny VLM 训练起来，让你看到训练前后预测真的变了。

---

## 第一部分: toy driving 图像是什么

今天不用真实街景图片。

代码用 `common/driving_toy.py` 生成一个 64x64 的 BEV 小图。

BEV 是 bird's-eye view，也就是俯视图。

图里有道路、自车、前车、车道线，还有左转、右转或刹车提示。

为什么不用真实图?

因为今天的目标不是视觉效果，而是理解 VLM 的数据流。

真实图会引入很多干扰: 分辨率、归一化、预训练权重、下载模型、显存。

toy 图反而更好，因为它让每个 shape 都很清楚。

一张图进入模型前会变成:

```text
pixel_values: [B, 3, 64, 64]
```

这和真实 VLM 的第一步是同一个抽象。

真实 VLM 可能是 `[B, 3, 224, 224]`，但本质仍然是:

```text
image -> pixel tensor
```

---

## 第二部分: 01_patchify_image.py

第一个脚本是:

```text
day4_vlm/01_patchify_image.py
```

核心函数是:

```python
def patchify(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
```

输入是:

```text
pixel_values: [B, C, H, W]
```

在我们的 toy 例子里:

```text
[B, 3, 64, 64]
```

如果 `patch_size = 8`，那么 64x64 的图片每边能切成 8 块:

```text
64 / 8 = 8
```

总 patch 数量是:

```text
8 * 8 = 64
```

每个 patch 的原始维度是:

```text
C * patch_size * patch_size
= 3 * 8 * 8
= 192
```

所以输出是:

```text
patches: [B, 64, 192]
```

这一步非常像 tokenizer。

文本 tokenizer 做的是:

```text
字符串 -> token 序列
```

patchify 做的是:

```text
图片 -> patch 序列
```

注意，patch 还不是最终 visual token。

patch 只是图像小块摊平成的像素向量。

但是这一步已经把二维图片改造成 Transformer 擅长处理的序列形式。

脚本里还有一个 `unpatchify`。

它不是 VLM 必需模块，而是为了教学验证:

```text
patchify 只是重新排列像素，没有丢信息。
```

运行脚本时你会看到:

```text
patches: (1, 64, 192)
rebuild max error: 0.0
```

`rebuild max error = 0.0` 表示从 patches 拼回图片后，和原图完全一样。

所以第一节你只要记住一句话:

```text
图片先被切成 patch 序列，patches 的 shape 是 [B, N, patch_dim]。
```

---

## 第三部分: 02_tiny_vit_encoder.py

第二个脚本是:

```text
day4_vlm/02_tiny_vit_encoder.py
```

这一节把 patch 变成 visual features。

代码里有一个 `TinyViTEncoder`。

它的主线是:

```python
patches = patchify(pixel_values, self.patch_size)
patch_tokens = self.patch_projection(patches)
patch_tokens = patch_tokens + self.position_embedding
visual_features = self.encoder(patch_tokens)
```

第一步:

```python
self.patch_projection = nn.Linear(patch_dim, vision_dim)
```

这很像 Day2 的 token embedding。

区别是文本 token embedding 是查表:

```text
token id -> embedding vector
```

图像 patch embedding 是线性投影:

```text
patch pixels -> embedding vector
```

shape 从:

```text
[B, 64, 192]
```

变成:

```text
[B, 64, 64]
```

第二步:

```python
self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, vision_dim) * 0.02)
```

这和 GPT 的 position embedding 是同一个原因。

Transformer 本身不知道顺序。

对于图像来说，它也不知道哪个 patch 在左上角，哪个 patch 在右下角。

所以 patch token 必须加位置编码。

代码里真正融合内容和位置的是这一行:

```python
patch_tokens = patch_tokens + self.position_embedding
```

这里的 `+` 可以先按传统特征融合来理解。

我们已经有了内容特征:

```text
x_i = Linear(patch_i)
```

再给每个空间位置一个可训练的位置向量:

```text
p_i
```

相加之后:

```text
z_i = x_i + p_i
```

`x_i` 表示第 `i` 个 patch 里有什么，`p_i` 表示第 `i` 个 patch 在哪里，`z_i` 就表示这个位置上的这个 patch。

所以 `position_embedding` 可以被理解成一种可训练的位置特征偏移量。

同样的视觉纹理，如果出现在左上角和右下角，加上的 `p_i` 不同，进入 Transformer 的 token 也就不同。

理论上，绑定内容和位置不只有加法一种方式。

你也可以想象:

```text
concat:   [x_i; p_i]
MLP:      MLP([x_i; p_i])
gate:     x_i + g_i * p_i
multiply: x_i * p_i
```

这些方式都可能奏效。

但 ViT 和 GPT 一样通常用加法，因为它有三个实际好处:

```text
1. 不改变 hidden size，后面的 Transformer 不用改。
2. 参数少，训练稳定。
3. 后面的 attention 已经足够强，可以继续学习复杂交互。
```

注意，加法本身虽然简单，但后面马上会进入 attention。

如果:

```text
z_i = x_i + p_i
```

那么 attention 里会计算:

```text
q_i = W_Q z_i
k_i = W_K z_i
v_i = W_V z_i
```

也就是:

```text
q_i = W_Q x_i + W_Q p_i
```

后面再做:

```text
score(i, j) = q_i dot k_j
```

这个分数会同时受到内容和位置影响。

所以这里不是“简单相加以后就结束了”。

更准确地说:

```text
加法负责把内容和位置放进同一个 token；
attention 负责在这些带位置的 token 之间建模关系。
```

第三步:

```python
self.encoder = nn.TransformerEncoder(layer, num_layers=1)
```

这就是一个极小版 ViT encoder。

它让 patch token 之间通过 self-attention 通信。

输出叫:

```text
visual_features: [B, 64, 64]
```

运行脚本时你会看到:

```text
patches: (2, 64, 192)
patch_tokens after Linear + position: (2, 64, 64)
visual_features after Transformer: (2, 64, 64)
```

这里的关键不是数值大小，而是你要知道每一步为什么存在:

```text
patch_projection: 把像素 patch 变成 embedding
position_embedding: 告诉模型 patch 在哪里
TransformerEncoder: 让 patch 之间交换信息
```

---

## 第四部分: 03_minigpt4_bridge.py

第三个脚本是:

```text
day4_vlm/03_minigpt4_bridge.py
```

这是今天最重要的结构。

它出现三个模块:

```text
TinyViTEncoder
VisionProjector
TinyTextBackbone
```

`TinyViTEncoder` 负责看图。

`TinyTextBackbone` 负责处理 token 序列。

`VisionProjector` 负责桥接两个世界。

为什么需要 projector?

因为 vision encoder 输出的是:

```text
visual_features: [B, 64, 64]
```

最后一维是 `vision_dim = 64`。

而 tiny LLM 的 hidden size 是:

```text
llm_dim = 96
```

text embedding 是:

```text
text_tokens: [B, 6, 96]
```

如果你想把视觉 token 和文本 token 拼起来，最后一维必须一样。

所以 projector 做:

```text
[B, 64, 64] -> [B, 64, 96]
```

代码是:

```python
visual_tokens = projector(visual_features)
text_tokens = tiny_llm.token_embedding(input_ids)
vlm_inputs = torch.cat([visual_tokens, text_tokens], dim=1)
hidden = tiny_llm(vlm_inputs)
```

拼接后 shape 是:

```text
vlm_inputs: [B, 70, 96]
```

为什么是 70?

因为:

```text
64 个 visual tokens + 6 个 text tokens = 70
```

你可以把序列想成:

```text
[图像patch1, 图像patch2, ..., 图像patch64, BOS, what, affects, ego, planning, ANSWER]
```

这就是 VLM 的核心结构。

MiniGPT-4 的真实 projector 可能更复杂，但抽象仍然是:

```text
vision encoder -> projector -> LLM hidden space
```

---

## 第五部分: TinyTextBackbone 和 Day2 GPT 的关系

`TinyTextBackbone` 有:

```python
self.token_embedding
self.position_embedding
self.layers = nn.TransformerEncoder(...)
self.final_norm
```

它和 Day2 mini GPT 很像，但它不是 causal LM。

今天我们不是训练它生成一长段文字。

今天只是用融合后的 hidden state 做分类。

所以它更像一个 encoder-style backbone:

```text
visual tokens + text tokens -> hidden states
```

然后取 `ANSWER` 位置的 hidden state。

这件事你在 Day3 已经见过:

```text
取某个 token 位置的 hidden state。
```

今天取的是 ANSWER token 的 hidden。

因为我们希望这个位置聚合图像信息和问题信息，然后回答。

---

## 第六部分: 04_train_tiny_vlm.py

第四个脚本是:

```text
day4_vlm/04_train_tiny_vlm.py
```

它定义了真正可训练的 `TinyVLM`。

模型模块是:

```text
vision_encoder
projector
tiny_llm
answer_head
```

Dataset 每次返回:

```text
image: [3, 64, 64]
input_ids: [6]
label: reason id
```

prompt 是:

```text
BOS what affects ego planning ANSWER
```

label 来自 `reason_from_sample(sample)`。

这里要注意，label 不是简单的原始 command。

比如 command 是 keep，但如果前车很近并且相对速度为负，reason 仍然应该是:

```text
brake_due_to_close_lead
```

所以模型不能只背 command。它要从图像里看出前车状态和提示。

forward 主线是:

```python
visual_features, patch_tokens, patches = self.vision_encoder(pixel_values)
visual_tokens = self.projector(visual_features)
text_tokens = self.tiny_llm.token_embedding(input_ids)
vlm_inputs = torch.cat([visual_tokens, text_tokens], dim=1)
hidden = self.tiny_llm(vlm_inputs)
answer_position = visual_tokens.shape[1] + input_ids.shape[1] - 1
answer_hidden = hidden[:, answer_position]
logits = self.answer_head(answer_hidden)
```

最关键的是这一句:

```python
answer_position = visual_tokens.shape[1] + input_ids.shape[1] - 1
```

为什么?

因为总序列前面是 visual tokens，后面是 text tokens。

ANSWER 是 text tokens 的最后一个。

所以 ANSWER 在拼接后的位置是:

```text
visual token 数量 + text token 数量 - 1
```

在默认设置里:

```text
64 + 6 - 1 = 69
```

取出来:

```text
answer_hidden: [B, 96]
```

再过分类头:

```text
logits: [B, 4]
```

这 4 类是:

```text
keep
brake_due_to_close_lead
lane_change_left
lane_change_right
```

---

## 第七部分: 训练时看什么

运行:

```bash
.venv/bin/python day4_vlm/04_train_tiny_vlm.py
```

脚本会先打印 shape trace:

```text
pixel_values: (..., 3, 64, 64)
patches: (..., 64, 192)
patch_tokens: (..., 64, 64)
visual_features: (..., 64, 64)
visual_tokens: (..., 64, 96)
text_tokens: (..., 6, 96)
vlm_inputs: (..., 70, 96)
answer_hidden: (..., 96)
logits: (..., 4)
```

然后它会打印训练前预测。

训练前是随机初始化，所以预测通常很乱。

之后进入 Day1 那条训练闭环:

```text
logits -> cross entropy loss -> backward -> optimizer.step
```

每个 epoch 打印:

```text
train_loss
val_acc
```

最后打印训练后预测。

这一步非常重要。

如果只打印 shape，你只能知道代码没崩。

但训练前/训练后预测对比告诉你:

```text
这个 tiny VLM 真的学到了一点视觉到语义的映射。
```

这才是今天代码“发挥作用”的地方。

---

## 收尾

今天你要记住四句话。

第一:

```text
patchify 把图片变成 patch 序列。
```

第二:

```text
TinyViTEncoder 把 patch 序列变成 visual features。
```

第三:

```text
VisionProjector 把 vision_dim 对齐到 llm_dim。
```

第四:

```text
visual tokens + text tokens 进入 tiny LLM 后，可以取 ANSWER hidden 做任务。
```

明天 Day5，我们不再只问“结构怎么接”，而是问:

```text
projector 怎么训练? 为什么要 freeze? instruction tuning 到底调什么?
```
