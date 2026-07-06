
# Mini GPT

```text
Mini GPT 的整体结构是：

token embedding
position embedding
Transformer Block
LayerNorm
LM Head

token embedding 负责把 token id 变成向量。
输入 shape 是 [B, T]。
输出 shape 是 [B, T, C]。

position embedding 负责告诉模型每个 token 在序列中的位置。
因为 self-attention 本身不天然知道顺序。

token embedding 和 position embedding 相加：
x = tok_emb + pos_emb

Transformer Block 包含两部分：
1. causal self-attention
2. feed forward network

attention 负责 token 之间通信。
feed forward 负责每个 token 自己的非线性变换。

residual connection 是：
x = x + sublayer(x)

它的作用是让原始信息保留下来，也让梯度更容易传播。

LayerNorm 的作用是稳定训练。
常见 GPT 结构使用 Pre-LN：
x = x + attention(layernorm(x))
x = x + feedforward(layernorm(x))

LM Head 是最后一层线性层。
它把 hidden state 映射成 vocab_size 维 logits。

最终 logits shape 是：
[B, T, vocab_size]

训练目标还是 next-token prediction。
```