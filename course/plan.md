可以压缩，但先说实话：**一周内不可能真正“精通”这些知识**。你这一周的目标应该是：

> **把 DL / PyTorch / Transformer / VLM / VLA / OneVL 的关键链路跑通，形成可继续深入的工程骨架。**

也就是说，不追求“完整学完”，追求：

```text
能读懂主要代码
能改动小模块
能解释 hidden state / latent token / loss / prefill / decode
能跑通一个最小 VLA toy 项目
能开始读 OneVL
```

下面给你一套 **7 天冲刺项目制安排**。每天都有：

```text
学习资料
手写任务
repo 阅读任务
交付物
验收标准
```


> **Day 4 之后已重写为新版路线。** 请从 [`course/day4_to_day10_replan.md`](day4_to_day10_replan.md) 进入；旧 Day4-Day7 材料已删除。

---

# 总体原则

这一周不要平均学习。

你每天 5～8 小时，建议按这个比例：

| 内容                   | 时间占比 |
| ---------------------- | -------: |
| 手写代码               |      40% |
| 跑 repo / 改 repo      |      30% |
| 看 crash course / 文档 |      20% |
| 写总结图 / 笔记        |      10% |

每天最后必须留下一个产出：

```text
一个可运行脚本
一个结构图
一个 markdown 总结
一个实验结果
```

没有产出，等于没学。

---

# 第 0 天：准备环境，别在环境上死磕

如果你已经在 A100 服务器上有环境，可以跳过一半。但我建议你单独准备一个学习 repo。

## 目标

建立一个 `onevl_bootcamp` 文件夹，所有手写代码都放进去。

```bash
mkdir -p ~/onevl_bootcamp
cd ~/onevl_bootcamp

mkdir day1_pytorch day2_transformer day3_llm day4_vlm day5_vla_toy day6_onevl day7_experiment
```

## 环境建议

先用一个干净环境：

```bash
uv venv .venv --python 3.10
source .venv/bin/activate

uv pip install torch torchvision torchaudio
uv pip install transformers accelerate datasets pillow matplotlib numpy tqdm einops
uv pip install jupyter ipykernel
```

如果 A100 服务器 CUDA / PyTorch 已经有兼容版本，不要反复折腾最新版。能跑就先跑。

---

# Day 1：PyTorch / DL 最小闭环

## 今天要掌握什么

必须掌握：

```text
Tensor
shape
nn.Module
forward
loss
backward
optimizer.step
Dataset
DataLoader
train/eval
checkpoint
```

这一天是地基。你后面看 OneVL 的所有训练代码，都绕不开这些。

---

## 看什么

### 资料 1：PyTorch 60 Minute Blitz

看官方 PyTorch 的 **Deep Learning with PyTorch: A 60 Minute Blitz**。它覆盖 tensor、autograd、神经网络和一个小分类网络，是你这种有编程基础但 DL 不系统的人最适合的入口。([PyTorch Docs][1])

只看这些部分：

```text
Tensors
A Gentle Introduction to torch.autograd
Neural Networks
Training a Classifier
```

不要看太多扩展教程。

---

## 手写项目 1：MLP 拟合一个非线性函数

文件：

```text
day1_pytorch/01_mlp_regression.py
```

任务：

用 PyTorch 写一个 MLP，拟合下面这个函数：

```text
y = sin(x) + 0.3 * x
```

要求：

```text
1. 自己生成训练数据
2. 自己定义 nn.Module
3. 用 MSELoss
4. 用 AdamW
5. 每 100 step 打印 loss
6. 保存 loss 曲线
7. 保存 checkpoint
```

你必须在代码里打印 shape：

```python
print("x:", x.shape)
print("y:", y.shape)
print("pred:", pred.shape)
```

验收标准：

```text
loss 能稳定下降
能画出预测曲线和真实曲线
知道 [B, 1] 里面 B 和 1 分别是什么
```

---

## 手写项目 2：自己写 Dataset / DataLoader

文件：

```text
day1_pytorch/02_dataset_dataloader.py
```

任务：

把刚才的数据封装成：

```python
class SineDataset(torch.utils.data.Dataset):
    ...
```

要求：

```text
1. 实现 __len__
2. 实现 __getitem__
3. 用 DataLoader batch 训练
4. 尝试 batch_size = 4 / 32 / 256
5. 观察 loss 波动
```

验收标准：

你能解释：

```text
Dataset 返回单样本
DataLoader 组成 batch
batch 第一维是 B
```

---

## 手写项目 3：梯度流检查

文件：

```text
day1_pytorch/03_gradient_check.py
```

任务：

训练时打印：

```python
for name, p in model.named_parameters():
    if p.grad is not None:
        print(name, p.grad.norm())
```

然后分别测试：

```python
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

再测试：

```python
with torch.no_grad():
    pred = model(x)
```

和：

```python
hidden = hidden.detach()
```

验收标准：

你要能说清楚：

```text
no_grad：不记录计算图
detach：截断梯度
requires_grad：参数是否参与训练
```

这对后面理解 OneVL 的 Stage 0 / Stage 1 / Stage 2 很关键。

---

## 今天最终交付

你要有这 3 个脚本：

```text
day1_pytorch/01_mlp_regression.py
day1_pytorch/02_dataset_dataloader.py
day1_pytorch/03_gradient_check.py
```

再写一个总结：

```text
day1_pytorch/README.md
```

内容只写 5 个问题：

```text
1. Tensor shape 怎么看？
2. nn.Module 的 forward 做什么？
3. loss.backward() 做什么？
4. detach 和 no_grad 有什么区别？
5. Dataset 和 DataLoader 怎么分工？
```

---

# Day 2：Transformer / Attention / Causal LM

## 今天要掌握什么

必须掌握：

```text
token
embedding
Q/K/V
self-attention
causal mask
Transformer block
next-token prediction
logits
cross entropy
```

今天的目标不是跑大模型，而是**手写一个小 Transformer**。

---

## 看什么

### 资料 1：Karpathy build-nanogpt

用 Karpathy 的 `build-nanogpt`，它是从零复现 nanoGPT 的 step-by-step repo，适合你快速理解 GPT 结构。([GitHub][2])

### 资料 2：nanoGPT

`nanoGPT` 是一个很简洁的 GPT 训练/微调 repo，README 里说它是为了训练/微调中等规模 GPT 的简单快速实现。([GitHub][3])

这两个不要都完整啃。今天建议：

```text
先看 build-nanogpt 的代码组织
再看 nanoGPT 的 model.py / train.py
```

---

## 手写项目 1：字符级 tokenizer

文件：

```text
day2_transformer/01_char_tokenizer.py
```

任务：

给一段文本：

```text
"the car slows down and the ego vehicle should brake"
```

实现：

```python
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

encode(text)
decode(ids)
```

验收标准：

```text
文本可以 encode 成 token id
token id 可以 decode 回文本
```

你要明白：

```text
tokenizer 不是模型
tokenizer 是把离散符号变成 token id 的工具
```

---

## 手写项目 2：Bigram LM

文件：

```text
day2_transformer/02_bigram_lm.py
```

任务：

实现最简单的 next-token prediction：

```python
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
```

要求：

```text
1. 输入 idx shape: [B, T]
2. 输出 logits shape: [B, T, vocab_size]
3. 用 cross_entropy 训练
4. 写 generate 函数
```

验收标准：

你能解释：

```text
logits 是每个位置对下一个 token 的预测分布
cross entropy 是预测 token 分布和真实 token id 的差距
```

---

## 手写项目 3：Self-Attention

文件：

```text
day2_transformer/03_self_attention.py
```

实现一个最小 self-attention：

```python
q = self.query(x)
k = self.key(x)
v = self.value(x)

wei = q @ k.transpose(-2, -1) / math.sqrt(head_size)
wei = wei.masked_fill(mask == 0, float("-inf"))
wei = F.softmax(wei, dim=-1)

out = wei @ v
```

要求打印：

```text
x: [B, T, C]
q: [B, T, H]
k: [B, T, H]
attention score: [B, T, T]
out: [B, T, H]
```

验收标准：

你必须能解释：

```text
为什么 attention score 是 [B, T, T]
为什么 causal mask 能防止看未来
为什么 decode 只能逐 token
```

---

## 手写项目 4：迷你 GPT

文件：

```text
day2_transformer/04_mini_gpt.py
```

最小结构：

```text
token embedding
position embedding
N 个 Transformer Block
LayerNorm
LM Head
```

不用追求效果，能训练下降即可。

验收标准：

```text
loss 从随机水平下降
generate 能生成类似训练文本风格的字符
```

---

## 今天最终交付

```text
day2_transformer/01_char_tokenizer.py
day2_transformer/02_bigram_lm.py
day2_transformer/03_self_attention.py
day2_transformer/04_mini_gpt.py
```

再写：

```text
day2_transformer/README.md
```

回答：

```text
1. input_ids 是什么？
2. logits 是什么？
3. causal mask 为什么必要？
4. self-attention 的 [B,T,T] 是什么？
5. next-token prediction 如何训练？
```

---

# Day 3：HuggingFace / LLM / prefill-decode

## 今天要掌握什么

必须掌握：

```text
AutoTokenizer
AutoModelForCausalLM
input_ids
attention_mask
labels
generate
hidden_states
KV cache
prefill
decode
```

这是从你手写 mini GPT 过渡到真实 LLM。

---

## 看什么

### 资料 1：HuggingFace causal language modeling

HuggingFace 官方文档对 causal language modeling 的定义很清楚：它是预测 token 序列中的下一个 token，并且模型只能关注左侧已有 token。([Hugging Face][4])

### 资料 2：HuggingFace LLM Course

HuggingFace 课程里关于 causal LM 的章节也值得快速看，尤其是“从零训练 causal language model”这一节，用于理解训练和预训练/微调差别。([Hugging Face][5])

---

## 手写项目 1：跑一个小 Causal LM

文件：

```text
day3_llm/01_hf_causal_lm_infer.py
```

用一个小模型，比如：

```text
sshleifer/tiny-gpt2
```

任务：

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
```

要求：

```text
1. tokenizer(prompt)
2. 打印 input_ids shape
3. model(**inputs)
4. 打印 logits shape
5. model.generate(...)
6. decode 输出
```

验收标准：

你能解释：

```text
input_ids: [B, T]
logits: [B, T, vocab_size]
generate 是自回归生成
```

---

## 手写项目 2：取 hidden_states

文件：

```text
day3_llm/02_hidden_states_probe.py
```

任务：

```python
outputs = model(
    **inputs,
    output_hidden_states=True,
    return_dict=True
)
```

打印：

```python
len(outputs.hidden_states)
outputs.hidden_states[-1].shape
```

然后取最后一个 token 的 hidden state：

```python
last_hidden = outputs.hidden_states[-1][:, -1, :]
```

验收标准：

你能解释：

```text
hidden state 是每个 token 在每一层后的内部表示
最后一层 hidden state 可以接不同 head 做任务
```

这一步直接对应 OneVL：

```text
取 latent token 位置 hidden state
送入 auxiliary decoder
```

---

## 手写项目 3：模拟 latent token

文件：

```text
day3_llm/03_fake_latent_token.py
```

任务：

给 tokenizer 添加特殊 token：

```python
special_tokens = {
    "additional_special_tokens": [
        "<latent_1>",
        "<latent_2>",
        "<latent_3>",
        "<latent_4>"
    ]
}
tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
```

构造 prompt：

```text
Scene: front vehicle is slowing down.
Reasoning tokens: <latent_1><latent_2><latent_3><latent_4>
Answer:
```

要求：

```text
1. 找到 latent token 的 input_id
2. 找到它们在 sequence 中的位置
3. 取这些位置的 hidden_states
4. 打印 shape: [B, num_latent, hidden_dim]
```

验收标准：

你能解释：

```text
latent token 本身只是特殊 token
真正有信息的是该位置经过 Transformer 后的 hidden state
```

这就是 OneVL 最关键的抽象。

---

## 手写项目 4：prefill / decode 粗测

文件：

```text
day3_llm/04_prefill_decode_timing.py
```

任务：

比较两种 prompt：

```text
A：短 prompt + 生成长 CoT
B：长 prompt 中预填 latent tokens + 只生成短 answer
```

你不用严格测底层 CUDA，只要用 `time.perf_counter()` 比较 generate 时间。

验收标准：

你能解释：

```text
长文本 CoT 如果在 decode 阶段生成，会逐 token 变慢
latent token 如果作为输入 prefill，处理方式不同
```

---

## 今天最终交付

```text
day3_llm/01_hf_causal_lm_infer.py
day3_llm/02_hidden_states_probe.py
day3_llm/03_fake_latent_token.py
day3_llm/04_prefill_decode_timing.py
```

README 回答：

```text
1. input_ids / attention_mask / logits 分别是什么？
2. generate 和 forward 有什么区别？
3. hidden_states 怎么取？
4. latent token position 怎么找？
5. prefill 和 decode 为什么速度不同？
```

---

# Day 4：VLM / 图像如何进入 LLM

## 今天要掌握什么

必须掌握：

```text
image processor
pixel_values
vision encoder
projector
visual tokens
multimodal prompt
VLM generate
```

今天先手写一个 tiny VLM 核心链路，再跑真实 VLM 推理和结构理解。

---

## 看什么

### 资料 1：Qwen3-VL 文档

HuggingFace 的 Qwen3-VL 文档说明 Qwen3-VL 模型是 PyTorch `nn.Module` 子类，可以像普通 PyTorch Module 一样使用。([Hugging Face][6])

### 资料 2：Qwen3-VL 官方 repo / model card

Qwen3-VL 官方介绍强调它是 Qwen 系列的视觉语言模型，支持图像、视频和文本等多模态输入。([GitHub][7])

### 资料 3：LLaVA repo

LLaVA 是经典视觉指令微调路线，repo 介绍其目标是用视觉指令微调构建 large language and vision assistant。([GitHub][8])

你今天优先用 **Qwen3-VL 或 Qwen2.5-VL 小模型** 跑。LLaVA 只用来理解典型 VLM 结构：

```text
vision encoder
projector
LLM
```

---

## 手写项目 1：手写 tiny VLM 核心链路

文件：

```text
day4_vlm/04_tiny_vlm_from_scratch.py
```

任务：

从零实现：

```text
pixel_values -> patchify -> vision encoder -> projector -> visual tokens
input_ids -> text embeddings -> text tokens
visual tokens + text tokens -> multimodal Transformer -> answer hidden
```

验收标准：

```text
能打印 image_patches / visual_features / visual_tokens / text_tokens / multimodal_tokens 的 shape
能解释 projector 为什么存在
能说明 VLM 不是把图片先 caption 成文字再交给 LLM
```

---

## 手写项目 2：跑 Qwen-VL / 小 VLM 推理

文件：

```text
day4_vlm/01_vlm_infer.py
```

任务：

输入一张道路图片，问：

```text
Describe the driving scene. Is there any object that may affect ego vehicle planning?
```

要求打印：

```text
processor 输出有哪些 key
input_ids shape
pixel_values shape
generate 输出
```

验收标准：

你能解释：

```text
文本走 tokenizer
图像走 image processor
最后一起进入模型
```

---

## 手写项目 3：构造自动驾驶 prompt

文件：

```text
day4_vlm/02_driving_prompt.py
```

构造类似 OneVL 的 prompt：

```text
You are an autonomous driving planning assistant.

Ego state:
- speed: 8.5 m/s
- acceleration: -0.2 m/s^2

Navigation command:
- go straight

History trajectory:
- t-1.5: (-4.0, 0.1)
- t-1.0: (-2.8, 0.1)
- t-0.5: (-1.4, 0.0)

Question:
Analyze the scene and predict the safe future trajectory.
```

任务：

```text
1. 跑 VLM 生成场景解释
2. 不要求轨迹准确
3. 重点观察 prompt 如何组织
```

验收标准：

你能解释：

```text
ego state / command / history trajectory 可以先文本化
VLM 不一定天然懂轨迹，但能理解 prompt
```

---

## 手写项目 4：VLM hidden state probe

文件：

```text
day4_vlm/03_vlm_hidden_probe.py
```

任务：

如果模型支持 `output_hidden_states=True`，取最后层 hidden state：

```text
outputs.hidden_states[-1]
```

找出最后几个 text token 的 hidden state。

如果 Qwen-VL 结构太复杂，你至少要打印 `model` 结构：

```python
print(model)
```

并定位：

```text
vision model
projector / merger
language model
lm_head
```

验收标准：

你能在模型结构里指出：

```text
图像从哪里进
文本从哪里进
最后在哪里生成 token
```

---

## 今天最终交付

```text
day4_vlm/04_tiny_vlm_from_scratch.py
day4_vlm/01_vlm_infer.py
day4_vlm/02_driving_prompt.py
day4_vlm/03_vlm_hidden_probe.py
```

README 回答：

```text
1. pixel_values 是什么？
2. image processor 和 tokenizer 分别做什么？
3. vision encoder 和 LLM 如何连接？
4. VLM 和 LLM 的输入差别是什么？
5. 为什么 VLM 可以作为 VLA backbone？
```

---

# Day 5：手写一个最小 VLA toy 项目

## 今天要掌握什么

必须掌握：

```text
VLA 输入输出
trajectory waypoint
状态/指令/历史轨迹如何编码
trajectory loss
latent action head
像素 + 语言/状态到轨迹
可视化轨迹
```

今天你要写一个小型“伪 OneVL”训练任务。不要追求真实自动驾驶，只追求理解 VLA 输出 action 的机制。今天最后要把 Day 4 的视觉 token 接回来，做一个最小 pixel-language-action 模型。

---

## 项目：Toy Driving VLA

目录：

```text
day5_vla_toy/
```

你要实现：

```text
输入：
  ego speed
  lead car distance
  lead car relative speed
  command: keep / brake / left / right
  history trajectory

输出：
  future 6 个 waypoint
```

---

## 数据生成规则

文件：

```text
day5_vla_toy/01_generate_data.py
```

生成 synthetic 数据：

```python
sample = {
    "ego_speed": 8.0,
    "lead_distance": 20.0,
    "rel_speed": -3.0,
    "command": "brake",
    "history": [[-3, 0], [-2, 0], [-1, 0]],
    "future": [[1, 0], [1.8, 0], [2.4, 0], ...]
}
```

规则简单即可：

```text
command = keep：
  x 均匀增加，y 保持 0

command = brake：
  x 增长逐渐变慢

command = left：
  x 增加，y 逐渐增大

command = right：
  x 增加，y 逐渐减小
```

再加入前车影响：

```text
lead_distance 小 + rel_speed 负：
  未来 x 增长减小
```

验收标准：

```text
能生成 10000 条 jsonl
能画出几条 future trajectory
```

---

## 模型 1：MLP VLA baseline

文件：

```text
day5_vla_toy/02_mlp_vla.py
```

输入编码：

```text
ego_speed
lead_distance
rel_speed
command one-hot
history flatten
```

输出：

```text
future 6 个点 → 12 个数
```

loss：

```text
MSELoss(pred_future, gt_future)
```

验收标准：

```text
loss 下降
预测轨迹能大致拟合 keep/brake/left/right
```

---

## 模型 2：Token-based VLA

文件：

```text
day5_vla_toy/03_token_vla.py
```

这一步更接近 OneVL 思想。

把输入文本化：

```text
ego speed is 8.0, lead distance is 20.0, command is brake
```

把 command / 数值离散成 token id。

模型可以简单点：

```text
Embedding
Transformer Encoder 或 GRU
MLP head 输出轨迹
```

重点是理解：

```text
文本/状态 token → hidden state → trajectory head
```

验收标准：

```text
能解释 token 输入如何变成轨迹输出
```

---

## 模型 3：加入 fake latent token

文件：

```text
day5_vla_toy/04_latent_vla.py
```

构造输入：

```text
[state tokens] [command tokens] [history tokens] [LATENT1] [LATENT2] [LATENT3] [LATENT4]
```

取 latent 位置 hidden state：

```python
latent_h = hidden[:, latent_positions, :]
latent_summary = latent_h.mean(dim=1)
traj = traj_head(latent_summary)
```

验收标准：

你必须能解释：

```text
latent token hidden state 被 trajectory head 使用
这和 OneVL 取 latent hidden state 给 decoder / trajectory 使用是同一个思想
```

---

## 模型 4：tiny pixel-language-action VLA

文件：

```text
day5_vla_toy/06_tiny_vla_from_pixels.py
```

任务：

把 toy sample 渲染成一张 BEV 图像，并训练：

```text
BEV image -> visual tokens
state / command -> text tokens
visual tokens + text tokens + latent tokens -> Transformer
latent hidden -> trajectory head
```

验收标准：

```text
能打印 visual_tokens / text_tokens / latent_tokens / latent_h / trajectory 的 shape
能解释 VLM backbone 如何接 action head 变成 VLA
```

---

---

## 可视化

文件：

```text
day5_vla_toy/05_plot_trajectory.py
```

画：

```text
GT trajectory
Pred trajectory
history trajectory
```

不同 command 分别保存图片。

验收标准：

```text
你能肉眼判断模型是否学会 brake / left / right
```

---

## 今天最终交付

```text
day5_vla_toy/01_generate_data.py
day5_vla_toy/02_mlp_vla.py
day5_vla_toy/03_token_vla.py
day5_vla_toy/04_latent_vla.py
day5_vla_toy/05_plot_trajectory.py
```

README 回答：

```text
1. VLA 里的 action 在自动驾驶中是什么？
2. waypoint 如何表示？
3. MLP 输出轨迹和 token-based 输出轨迹有什么区别？
4. latent token 如何参与轨迹预测？
5. 为什么 trajectory loss 可以训练 latent hidden state？
```

这一天非常重要。你如果只看教程不写这个 toy 项目，后面 OneVL 很容易看成玄学。

---

# Day 6：OneVL repo 结构阅读 + 单样本推理

## 今天要掌握什么

必须掌握：

```text
OneVL repo 怎么组织
infer 脚本从哪里进
数据样本长什么样
latent token 参数在哪里
模型如何加载
输出如何解析
```

---

## 看什么

### 资料 1：OneVL GitHub

OneVL repo 说明它是自动驾驶 VLA 框架，目标是在保持 answer-only AR 模型推理延迟的同时获得高轨迹预测精度；它通过 dual-modal auxiliary decoders 监督 compact latent tokens 同时编码语言推理和未来场景动态。([GitHub][9])

### 资料 2：OneVL Project Page

项目页强调 OneVL 使用 language decoder 重建文本 CoT，并使用 visual world model decoder 预测未来帧 token，从而让 latent space 内化 causal scene dynamics。([Xiaomi-Embodied-Intelligence][10])

### 资料 3：OneVL 论文

论文页面确认其 GitHub 和项目页，并描述 OneVL 面向自动驾驶 one-step latent reasoning and planning。([arXiv][11])

---

## 任务 1：clone repo

```bash
cd ~/onevl_bootcamp/day6_onevl
git clone https://github.com/xiaomi-research/OneVL.git
cd OneVL
```

然后只做阅读，不要一上来跑完整 benchmark。

先找：

```bash
find . -maxdepth 2 -type f | sort
```

重点定位：

```text
README
infer_onevl.py
scripts/
test_data/
visual_tokenizer/
vq_decoder/
requirements
```

---

## 任务 2：画 repo 地图

文件：

```text
day6_onevl/repo_map.md
```

写清楚：

```text
1. 推理入口在哪里？
2. 数据样本在哪里？
3. 模型加载在哪里？
4. latent 参数在哪里？
5. benchmark 脚本在哪里？
6. visual tokenizer / visual decoder 相关代码在哪里？
```

验收标准：

你不用全懂，但必须知道“想看某件事应该去哪个文件”。

---

## 任务 3：读推理脚本

重点读：

```text
infer_onevl.py
scripts/run_infer*.sh
scripts/infer_navsim*.sh
```

找这些关键词：

```text
num_latent
num_latent_vis
model_path
processor
tokenizer
generate
hidden_states
trajectory
```

你要做一个表：

| 关键词         | 文件位置 | 作用                 |
| -------------- | -------- | -------------------- |
| num_latent     | ?        | language latent 数量 |
| num_latent_vis | ?        | visual latent 数量   |
| model_path     | ?        | checkpoint           |
| generate       | ?        | 输出轨迹/回答        |
| test_data      | ?        | 样本输入             |

---

## 任务 4：跑官方最小推理

先按 README 的最小样例跑。如果完整模型太大或需要下载慢，就先只跑 test data / mock 模式，至少让脚本走到数据加载和 prompt 构造。

你今天目标不是 benchmark 分数，而是：

```text
数据能读
prompt 能构造
模型能加载或至少知道卡在哪里
输出格式能看
```

如果卡在模型下载 / 权限 / 显存，记录下来，不要无限死磕。

---

## 任务 5：对照你的 toy latent VLA

你要写一个对应关系：

```text
你的 toy 项目：
[state tokens] [latent tokens] → trajectory head

OneVL：
[image tokens] [state/prompt tokens] [visual latent] [language latent] → trajectory output
```

再写：

```text
你的 toy latent：
直接用 MLP head 输出轨迹

OneVL latent：
还要送入 language decoder / visual decoder 做辅助监督
```

---

## 今天最终交付

```text
day6_onevl/repo_map.md
day6_onevl/infer_reading_notes.md
day6_onevl/onevl_vs_toy.md
```

验收标准：

你能回答：

```text
1. OneVL 的推理入口在哪里？
2. latent token 数量在哪里配置？
3. 输入样本格式长什么样？
4. 推理时 decoder 是否参与？
5. visual decoder 训练时到底监督什么？
```

---

# Day 7：小实验闭环：latent / prompt / trajectory / failure case

## 今天要掌握什么

必须掌握：

```text
怎么从“跑通”进入“实验”
怎么记录 failure case
怎么设计 ablation
怎么形成发现问题 → 改动 → 观察 的闭环
```

这是你后面项目能不能写成经历的关键。

---

## 任务 1：Toy VLA ablation

基于 Day 5 的 toy latent VLA，做 4 组实验：

| 实验 | 改动                                         |
| ---- | -------------------------------------------- |
| A    | 不使用 latent token，直接最后 token 输出轨迹 |
| B    | 使用 2 个 latent token                       |
| C    | 使用 4 个 latent token                       |
| D    | 使用 8 个 latent token                       |

记录：

```text
train loss
val loss
不同 command 下轨迹图
是否 brake 更稳定
```

验收标准：

你要能说：

```text
latent token 数量改变后，模型容量 / 信息瓶颈 / 训练稳定性有什么变化
```

这能帮助你理解 OneVL 为什么关心 latent 数量。

---

## 任务 2：加入 language auxiliary decoder toy

文件：

```text
day7_experiment/01_latent_with_language_aux.py
```

在 toy 项目里加一个辅助任务。

输入同样是：

```text
state + command + latent tokens
```

主任务：

```text
latent hidden → trajectory
```

辅助任务：

```text
latent hidden → classify command / risk_reason
```

比如构造标签：

```text
reason = "keep"
reason = "brake_due_to_close_lead"
reason = "lane_change_left"
reason = "lane_change_right"
```

loss：

```python
loss = traj_loss + 0.2 * reason_loss
```

验收标准：

你能解释：

```text
辅助任务不是为了最终输出 reason
而是为了让 latent hidden state 更有语义结构
```

这对应 OneVL 的 language auxiliary decoder。

---

## 任务 3：加入 visual/world auxiliary toy

文件：

```text
day7_experiment/02_latent_with_world_aux.py
```

构造一个“未来状态预测”辅助任务：

```text
当前 lead_distance, rel_speed
  → 预测 1 秒后 lead_distance
```

主任务：

```text
输出 ego future trajectory
```

辅助任务：

```text
latent hidden → future lead_distance
```

loss：

```python
loss = traj_loss + 0.2 * future_state_loss
```

这不是图像未来帧，但思想相同：

```text
用未来状态预测，逼 latent 学 dynamics
```

验收标准：

你能解释：

```text
world auxiliary loss 让 latent 不只学 command 语义，还学未来变化
```

这对应 OneVL 的 visual world model decoder。

---

## 任务 4：写 OneVL 小实验设计文档

文件：

```text
day7_experiment/onevl_experiment_plan.md
```

写 3 个你下一周可以真的在 OneVL 上做的实验。

建议：

### 实验 1：latent 数量 ablation

```text
num_latent / num_latent_vis 改变
观察 trajectory 结果、延迟、失败样本
```

### 实验 2：prompt ablation

```text
去掉 ego state
去掉 history trajectory
去掉 navigation command
观察输出变化
```

### 实验 3：failure case 分类

```text
前车减速
弯道
施工区
遮挡
横穿目标
```

每类保存：

```text
输入图像
prompt
输出轨迹
模型解释
错误类型
猜测原因
下一步改动
```

---

## 今天最终交付

```text
day7_experiment/01_latent_with_language_aux.py
day7_experiment/02_latent_with_world_aux.py
day7_experiment/onevl_experiment_plan.md
```

验收标准：

你能完整讲出：

```text
OneVL 为什么不是简单 VLM 输出轨迹
为什么 latent token 是瓶颈
为什么 language aux 和 visual aux 都有必要
为什么推理时 decoder 可以丢掉
```

---

# 一周总交付物清单

到第 7 天结束，你应该有这些东西：

```text
onevl_bootcamp/
  day1_pytorch/
    01_mlp_regression.py
    02_dataset_dataloader.py
    03_gradient_check.py
    README.md

  day2_transformer/
    01_char_tokenizer.py
    02_bigram_lm.py
    03_self_attention.py
    04_mini_gpt.py
    README.md

  day3_llm/
    01_hf_causal_lm_infer.py
    02_hidden_states_probe.py
    03_fake_latent_token.py
    04_prefill_decode_timing.py
    README.md

  day4_vlm/
    01_vlm_infer.py
    02_driving_prompt.py
    03_vlm_hidden_probe.py
    README.md

  day5_vla_toy/
    01_generate_data.py
    02_mlp_vla.py
    03_token_vla.py
    04_latent_vla.py
    05_plot_trajectory.py
    README.md

  day6_onevl/
    repo_map.md
    infer_reading_notes.md
    onevl_vs_toy.md

  day7_experiment/
    01_latent_with_language_aux.py
    02_latent_with_world_aux.py
    onevl_experiment_plan.md
```

这套东西比你看 50 小时网课有用。

---

# 你每天的时间分配模板

每天 5～8 小时，按这个来：

## 5 小时版本

```text
1 小时：看资料
2.5 小时：写代码
1 小时：跑实验 / debug
0.5 小时：写 README 总结
```

## 8 小时版本

```text
1.5 小时：看资料
4 小时：写代码
1.5 小时：跑实验 / debug
1 小时：写总结图 / README
```

不要超过 30% 时间看视频。你现在不是学生，是要快速形成工程能力。

---

# 每天必须问自己的 5 个问题

每天结束前回答：

```text
1. 今天我手写了什么？
2. 今天我跑通了什么？
3. 今天我打印了哪些 tensor shape？
4. 今天哪个概念能对应到 OneVL？
5. 明天我可以基于今天的代码改什么？
```

尤其是第 3 个：

> **不打印 shape，不算学 PyTorch / Transformer。**

---

# 最短路径总结

你这一周真正要打通的是这条链：

```text
Day 1:
Tensor / loss / backward / Dataset

Day 2:
token / attention / causal LM

Day 3:
HuggingFace / hidden_states / fake latent token

Day 4:
VLM / image token / visual-language prompt

Day 5:
VLA toy / trajectory output / latent hidden to trajectory

Day 6:
OneVL repo / infer pipeline / latent config

Day 7:
auxiliary loss / world loss / ablation / failure case
```

到这里，你还不是 DL 专家，但你已经能进入 OneVL 主线了。下一步就不是再泛学网课，而是围绕 OneVL 做：

```text
读论文图
跑 infer
看 dataset
找 latent token
看 decoder
做 ablation
写 failure case
```

这才是你要的 VLA / 世界模型能力闭环。

[1]: https://docs.pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html?utm_source=chatgpt.com "Deep Learning with PyTorch: A 60 Minute Blitz"
[2]: https://github.com/karpathy/build-nanogpt?utm_source=chatgpt.com "Video+code lecture on building nanoGPT from scratch"
[3]: https://github.com/karpathy/nanogpt?utm_source=chatgpt.com "karpathy/nanoGPT: The simplest, fastest repository ..."
[4]: https://huggingface.co/docs/transformers/tasks/language_modeling?utm_source=chatgpt.com "Causal language modeling"
[5]: https://huggingface.co/learn/llm-course/en/chapter7/6?utm_source=chatgpt.com "Training a causal language model from scratch"
[6]: https://huggingface.co/docs/transformers/en/model_doc/qwen3_vl?utm_source=chatgpt.com "Qwen3-VL"
[7]: https://github.com/qwenlm/qwen3-vl?utm_source=chatgpt.com "Qwen3-VL is the multimodal large language model ..."
[8]: https://github.com/haotian-liu/llava?utm_source=chatgpt.com "haotian-liu/LLaVA: [NeurIPS'23 Oral] Visual Instruction ..."
[9]: https://github.com/xiaomi-research/onevl?utm_source=chatgpt.com "xiaomi-research/onevl"
[10]: https://xiaomi-embodied-intelligence.github.io/OneVL/?utm_source=chatgpt.com "OneVL: One-Step Latent Reasoning and Planning with ..."
[11]: https://arxiv.org/html/2604.18486v3?utm_source=chatgpt.com "Xiaomi OneVL: One-Step Latent Reasoning and Planning ..."
