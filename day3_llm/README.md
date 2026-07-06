# Day 3：HuggingFace / LLM / prefill-decode

## 运行方式

```bash
mkdir -p day3_llm
cd day3_llm

uv pip install torch transformers accelerate

python 01_hf_causal_lm_infer.py
python 02_hidden_states_probe.py
python 03_fake_latent_token.py
python 04_prefill_decode_timing.py
```

如果第一次运行，会从 HuggingFace 下载 `sshleifer/tiny-gpt2`。

---

# 1. input_ids / attention_mask / logits 分别是什么？

## input_ids

`input_ids` 是 tokenizer 把文本切分成 token 后，再把每个 token 映射成整数 id 的结果。

形状：

```text
[B, T]
```

含义：

```text
B = batch size
T = sequence length，也就是 token 数量
```

模型不能直接吃字符串，只能吃 token id。

---

## attention_mask

`attention_mask` 表示哪些 token 是有效 token，哪些 token 是 padding。

通常：

```text
1 = 有效 token
0 = padding token
```

形状也是：

```text
[B, T]
```

如果一个 batch 里句子长短不同，短句会被 padding 补齐。attention_mask 告诉模型不要把 padding 当成真正文本。

---

## logits

`logits` 是模型输出的“下一个 token 预测分数”。

形状：

```text
[B, T, vocab_size]
```

含义：

```text
B = batch size
T = 输入 token 数量
vocab_size = tokenizer 词表大小
```

为什么是 `[B, T, vocab_size]`？

因为 causal LM 会在每一个位置都预测下一个 token。

例如输入：

```text
I love deep
```

模型会做类似这样的预测：

```text
I      -> love
love   -> deep
deep   -> learning
```

所以每个位置都有一个 vocab_size 维的预测分数。

生成下一个 token 时，最常用的是最后一个位置：

```python
next_token_logits = logits[:, -1, :]
```

---

# 2. labels 是什么？

`labels` 是训练 causal LM 时的监督目标。

在 HuggingFace 的 `AutoModelForCausalLM` 里，常见写法是：

```python
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
```

这看起来像是“输入自己预测自己”，但实际上模型内部会做 shift。

本质是：

```text
用第 i 个位置的输出，预测第 i+1 个 token。
```

例如：

```text
input_ids:  [I, love, deep, learning]
labels:     [I, love, deep, learning]
内部 shift 后：
I      -> love
love   -> deep
deep   -> learning
```

所以 labels 不是给 generate 用的，而是给训练 / loss 计算用的。

如果某些位置不想参与 loss，可以把 label 设为 `-100`。PyTorch 的 cross entropy 默认会忽略 `-100`。

---

# 3. forward 和 generate 有什么区别？

## forward

`forward` 是一次模型前向计算。

代码：

```python
outputs = model(**inputs)
```

输入：

```text
input_ids: [B, T]
attention_mask: [B, T]
```

输出：

```text
logits: [B, T, vocab_size]
```

如果打开：

```python
output_hidden_states=True
```

还能输出：

```text
hidden_states
```

forward 本身不会自动生成一长段文本，它只负责算一次。

---

## generate

`generate` 是完整的自回归生成流程。

代码：

```python
generated_ids = model.generate(**inputs, max_new_tokens=30)
```

它内部会循环：

```text
1. forward 当前 input_ids
2. 取最后一个位置 logits
3. 选出下一个 token
4. 把新 token 拼到 input_ids 后面
5. 继续 forward
6. 直到生成够 max_new_tokens 或遇到 eos token
```

所以：

```text
forward = 一次计算
generate = 多次 forward 组成的自回归生成循环
```

---

# 4. hidden_states 是什么？怎么取？

`hidden state` 是每个 token 经过 Transformer 某一层后的内部连续向量表示。

它不是 token id，也不是输出文字，而是模型内部表示。

取法：

```python
outputs = model(
    **inputs,
    output_hidden_states=True,
    return_dict=True,
)

hidden_states = outputs.hidden_states
```

`hidden_states` 通常是一个 tuple：

```text
hidden_states[0] = embedding 输出
hidden_states[1] = 第 1 层 Transformer 输出
hidden_states[2] = 第 2 层 Transformer 输出
...
hidden_states[-1] = 最后一层 Transformer 输出
```

最后一层 hidden state：

```python
hidden = outputs.hidden_states[-1]
```

形状：

```text
[B, T, hidden_dim]
```

取最后一个 token：

```python
last_token_hidden = hidden[:, -1, :]
```

形状：

```text
[B, hidden_dim]
```

它表示：最后一个 token 在读完整个前文后形成的内部表示。

---

# 5. latent token position 怎么找？

先添加特殊 token：

```python
special_tokens = {
    "additional_special_tokens": [
        "<latent_1>",
        "<latent_2>",
        "<latent_3>",
        "<latent_4>",
    ]
}

tokenizer.add_special_tokens(special_tokens)
model.resize_token_embeddings(len(tokenizer))
```

然后获得 latent token 对应的 token id：

```python
latent_token_ids = tokenizer.convert_tokens_to_ids([
    "<latent_1>",
    "<latent_2>",
    "<latent_3>",
    "<latent_4>",
])
```

再遍历 input_ids：

```python
positions = []

for pos, token_id in enumerate(input_ids[0].tolist()):
    if token_id in latent_token_ids:
        positions.append(pos)
```

这里要区分两个概念：

```text
token_id = 这个 token 在词表里的编号
position = 这个 token 在当前 sequence 里的位置
```

例如：

```text
<latent_1> 的 token_id 可能是 50257
它在当前 prompt 里可能出现在 position 9
```

最后取 hidden state：

```python
hidden = outputs.hidden_states[-1]
latent_hidden = hidden[:, positions, :]
```

形状：

```text
[B, num_latent, hidden_dim]
```

---

# 6. latent token 的本质是什么？

latent token 本身不是魔法。

它一开始只是 tokenizer 词表里的一个特殊符号：

```text
<latent_1>
```

进入模型前，它只是一个整数 id：

```text
50257
```

真正有信息的是：

```text
这个位置经过 embedding、self-attention、MLP、残差连接、LayerNorm 等 Transformer 计算之后得到的 hidden state。
```

所以重点不是：

```text
<latent_1> 这个字符串有什么语义
```

而是：

```text
<latent_1> 这个位置在上下文里经过 Transformer 处理后，形成了什么 hidden representation。
```

这就是 OneVL 里 latent token 的核心抽象：

```text
latent token position hidden state
    ↓
auxiliary decoder / trajectory decoder / planning head
```

也就是说：

```text
LLM 不只用来输出文字。
LLM 内部某些 token 位置的 hidden state，也可以作为下游模块的输入。
```

---

# 7. prefill / decode / KV cache 是什么？

LLM 推理可以粗略分成两个阶段：

```text
prefill
decode
```

---

## prefill

prefill 阶段处理完整输入 prompt。

例如：

```text
Scene: front vehicle is slowing down.
Reasoning tokens: <latent_1> <latent_2> <latent_3> <latent_4>
Answer:
```

这些输入 token 会被一次性送进模型。

prefill 阶段会建立初始 KV cache。

---

## decode

decode 阶段开始生成新 token。

decode 是逐 token 的：

```text
生成第 1 个 token
生成第 2 个 token
生成第 3 个 token
...
```

为什么不能一次性生成完？

因为第 2 个 token 依赖第 1 个 token，第 3 个 token 依赖第 2 个 token。

所以 decode 是自回归的。

---

## KV cache

Transformer attention 里每个 token 会产生：

```text
Query
Key
Value
```

生成新 token 时，历史 token 的 Key / Value 不需要重复算。

KV cache 就是把历史 token 的 Key / Value 存起来，后面继续复用。

所以：

```text
没有 KV cache：
每生成一个新 token，都要重复计算全部历史 token。

有 KV cache：
历史 token 的 Key / Value 复用，只计算新 token 相关部分。
```

KV cache 主要用于 inference，不用于正常训练。

---

# 8. 为什么长 CoT 慢？latent token 为什么不一样？

显式长 CoT 是输出文本：

```text
Let me think step by step...
First...
Second...
Third...
Therefore...
```

这些 token 都要在 decode 阶段一个一个生成。

所以：

```text
显式 CoT 越长，decode 步数越多，生成越慢。
```

latent token 如果是输入：

```text
<latent_1> <latent_2> <latent_3> <latent_4>
```

它们会在 prefill 阶段被处理。

所以对比是：

```text
显式长 CoT：
短 prompt + 长 decode

latent token：
较长 prompt/prefill + 短 decode
```

更准确地说：

```text
latent token 不是天然更快。
它的潜在价值是：用少量连续 hidden states 承载中间推理信息，减少显式文本 CoT 的 decode 长度。
```

---

# 9. 和 OneVL 的对应关系

今天这四个脚本对应 OneVL 的关键抽象：

```text
tokenizer / input_ids
    ↓
LLM forward
    ↓
hidden_states
    ↓
取 latent token positions
    ↓
latent hidden states
    ↓
auxiliary decoder / trajectory decoder / planning head
```

你今天不是在复现 OneVL，而是在练 OneVL 最底层的接口动作：

```python
latent_hidden = hidden[:, latent_positions, :]
```

只要你真正理解这一行，你就抓住了 OneVL latent token 机制的核心。

---

# 10. 今天必须背下来的版本

HuggingFace 里，tokenizer 负责把文本变成 input_ids 和 attention_mask。input_ids 的形状是 `[B, T]`，表示 batch 内每个样本的 token id 序列。attention_mask 的形状也是 `[B, T]`，表示哪些 token 是有效输入，哪些是 padding。

AutoModelForCausalLM 是用于 causal language modeling 的模型。它接收 input_ids，经过 embedding、Transformer blocks 和 LM head，输出 logits。logits 的形状是 `[B, T, vocab_size]`，表示每个位置对下一个 token 的预测分数。

forward 是一次前向计算；generate 是自回归生成循环。generate 会反复调用 forward，每次取最后一个位置的 logits，选出下一个 token，再把新 token 拼回输入序列继续生成。

labels 是训练 causal LM 时的监督目标，通常可以直接使用 input_ids。模型内部会 shift，使第 i 个位置预测第 i+1 个 token。labels 用于计算 loss，不是生成文本必须的输入。

如果调用模型时设置 output_hidden_states=True，就可以拿到每一层每个 token 的 hidden state。hidden_states[-1] 是最后一层 hidden state，形状是 `[B, T, hidden_dim]`。最后一个 token 的 hidden state 可以通过 `hidden[:, -1, :]` 取得。

latent token 本身只是特殊 token。它进入模型前只是一个 token id。真正有信息的是 latent token 所在位置经过 Transformer 处理后的 hidden state。我们可以先找到 latent token id，再在 input_ids 中找到它们的 sequence position，最后用 `hidden[:, positions, :]` 取出 latent hidden states，形状是 `[B, num_latent, hidden_dim]`。

LLM 推理分为 prefill 和 decode。prefill 一次性处理输入 prompt，并建立 KV cache。decode 逐 token 生成新 token，并复用 KV cache。KV cache 保存历史 token 在 attention 中的 key/value，避免每一步重复计算历史 token。长文本 CoT 如果作为输出生成，会增加 decode 步数；latent token 如果作为输入出现，则主要在 prefill 阶段被处理，可以减少显式文本推理的 decode 长度。

OneVL 的关键抽象是：不要只把 LLM 看成文字生成器，还要把某些 token 位置的 hidden state 看成可供下游 decoder 使用的连续表示。latent token 的核心不是 token 字符串，而是 latent token position 的 hidden state。
