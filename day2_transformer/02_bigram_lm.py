# day2_transformer/02_bigram_lm.py

"""
目标：
1. 实现最简单的 next-token prediction
2. 理解 logits
3. 理解 cross entropy
4. 理解 generate 为什么逐 token 生成

Bigram LM 的核心：
只根据当前 token 预测下一个 token。

例如：
输入:  t h e
目标:  h e 空格

idx:
[t, h, e]

targets:
[h, e, 空格]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# 为了结果可复现
torch.manual_seed(42)

# 简单训练文本
text = """
the car slows down and the ego vehicle should brake
the vehicle follows the lane and keeps safe distance
the front car stops and the ego car brakes slowly
the road is clear and the vehicle can accelerate
the ego vehicle should avoid collision
the traffic signal turns green and the car proceeds
the pedestrian crosses the street safely ahead
the speed limit is fifty kilometers per hour here
the weather is clear and visibility is good today
the engine starts smoothly and runs without issues
""" * 50

# 字符级 tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s: str) -> list[int]:
    return [stoi[ch] for ch in s]


def decode(ids: list[int]) -> str:
    return "".join([itos[i] for i in ids])


# 把整个文本编码成 token id
data = torch.tensor(encode(text), dtype=torch.long)

# 训练参数
batch_size = 16
block_size = 16
max_iters = 1000
eval_interval = 200
learning_rate = 1e-2

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_batch():
    """
    随机取一批训练数据。

    x 是输入：
    [B, T]

    y 是目标：
    [B, T]

    y 相当于 x 整体向右移动一位。
    模型看到 x[i]，要预测 y[i]。
    """
    ix = torch.randint(0, len(data) - block_size - 1, (batch_size,))

    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


class BigramLanguageModel(nn.Module):
    """
    最简单的语言模型。

    self.token_embedding_table:
    shape = [vocab_size, vocab_size]

    输入 idx:
    shape = [B, T]

    输出 logits:
    shape = [B, T, vocab_size]
    """

    def __init__(self, vocab_size: int):
        super().__init__()

        # 每个 token id 直接查出一个长度为 vocab_size 的向量
        # 这个向量就是对下一个 token 的 logits
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx:
        [B, T]

        logits:
        [B, T, vocab_size]

        targets:
        [B, T]
        """

        logits = self.token_embedding_table(idx)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # cross_entropy 需要：
            # logits:  [N, C]
            # targets: [N]
            #
            # 所以把 B 和 T 合并
            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """
        自回归生成。

        idx:
        [B, T]

        每次：
        1. 用当前 idx 预测 logits
        2. 只取最后一个位置的 logits
        3. softmax 得到概率
        4. 采样下一个 token
        5. 拼回 idx
        """

        for _ in range(max_new_tokens):
            logits, loss = self(idx)

            # 只关心最后一个 token 对下一个 token 的预测
            logits = logits[:, -1, :]  # [B, vocab_size]

            probs = F.softmax(logits, dim=-1)  # [B, vocab_size]

            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]

            idx = torch.cat((idx, idx_next), dim=1)  # [B, T+1]

        return idx


if __name__ == "__main__":
    print("device:", device)
    print("vocab_size:", vocab_size)
    print("chars:", chars)
    print()

    model = BigramLanguageModel(vocab_size).to(device)

    # 看一下初始 shape
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    print("检查 shape:")
    print("xb.shape:", xb.shape)
    print("yb.shape:", yb.shape)
    print("logits.shape:", logits.shape)
    print("loss:", loss.item())
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):
        xb, yb = get_batch()

        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0:
            print(f"step {step:4d}, loss {loss.item():.4f}")

    print()
    print("生成文本:")

    # 从 token 0 开始生成
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_ids = model.generate(context, max_new_tokens=300)[0].tolist()

    print(decode(generated_ids))