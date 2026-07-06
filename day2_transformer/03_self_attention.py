# day2_transformer/03_self_attention.py

"""
目标：
1. 理解 Q / K / V
2. 理解 attention score 为什么是 [B, T, T]
3. 理解 causal mask 为什么能防止看未来
4. 理解 self-attention 的输出 shape

Self-Attention 核心公式：

q = Wq(x)
k = Wk(x)
v = Wv(x)

score = q @ k.T / sqrt(head_size)
score = causal_mask(score)
weight = softmax(score)
out = weight @ v
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)


class SelfAttentionHead(nn.Module):
    """
    单头 causal self-attention。

    输入:
    x: [B, T, C]

    输出:
    out: [B, T, H]

    其中：
    B = batch size
    T = sequence length
    C = embedding dimension
    H = head size
    """

    def __init__(self, n_embd: int, head_size: int, block_size: int):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # 下三角矩阵，用于 causal mask
        # shape: [block_size, block_size]
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x, verbose=False):
        B, T, C = x.shape

        q = self.query(x)  # [B, T, H]
        k = self.key(x)    # [B, T, H]
        v = self.value(x)  # [B, T, H]

        H = q.shape[-1]

        # attention score
        # [B, T, H] @ [B, H, T] -> [B, T, T]
        wei = q @ k.transpose(-2, -1) / math.sqrt(H)

        # causal mask
        # 只保留下三角，未来位置填 -inf
        mask = self.tril[:T, :T]
        wei_masked = wei.masked_fill(mask == 0, float("-inf"))

        # softmax 后，每一行和为 1
        # 被 mask 的未来位置概率会变成 0
        att = F.softmax(wei_masked, dim=-1)

        # 加权汇总 value
        # [B, T, T] @ [B, T, H] -> [B, T, H]
        out = att @ v

        if verbose:
            print("x.shape:       ", x.shape)
            print("q.shape:       ", q.shape)
            print("k.shape:       ", k.shape)
            print("v.shape:       ", v.shape)
            print("score.shape:   ", wei.shape)
            print("mask.shape:    ", mask.shape)
            print("att.shape:     ", att.shape)
            print("out.shape:     ", out.shape)
            print()

            print("causal mask:")
            print(mask)
            print()

            print("第一条样本的 attention weight:")
            print(att[0])
            print()

        return out


if __name__ == "__main__":
    B = 2
    T = 4
    C = 8
    H = 4
    block_size = 8

    x = torch.randn(B, T, C)

    head = SelfAttentionHead(
        n_embd=C,
        head_size=H,
        block_size=block_size
    )

    out = head(x, verbose=True)