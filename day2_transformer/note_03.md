
# Self-Attention

```text
Self-attention 的作用：
让每个 token 根据上下文重新组织自己的表示。

Q 是 query：
当前 token 想找什么信息。

K 是 key：
每个 token 提供什么匹配线索。

V 是 value：
如果某个 token 被关注，它实际贡献什么内容。

q 的 shape 是 [B, T, H]。
k 的 shape 是 [B, T, H]。
v 的 shape 是 [B, T, H]。

attention score = q @ k.transpose(-2, -1)

所以：
[B, T, H] @ [B, H, T] = [B, T, T]

[B, T, T] 表示：
每个 token 对序列中每个 token 的关注分数。

第 i 行表示：
第 i 个 token 看其他 token 的权重。

causal mask 是下三角矩阵。
它保证第 i 个 token 只能看第 0 到第 i 个 token。
不能看 i 后面的未来 token。

mask 的位置填 -inf。
softmax 之后，-inf 对应的概率变成 0。

GPT 必须用 causal mask。
因为 GPT 的训练目标是 next-token prediction。
如果当前位置能看到未来 token，就等于提前看答案。
```