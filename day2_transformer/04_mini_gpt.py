# day2_transformer/04_mini_gpt.py

"""
目标：
手写一个最小 GPT。

结构：

token embedding
position embedding
N 个 Transformer Block
LayerNorm
LM Head

每个 Transformer Block：

x = x + causal_self_attention(layernorm(x))
x = x + feed_forward(layernorm(x))

训练目标：

给定 input_ids:
[B, T]

预测 targets:
[B, T]

输出 logits:
[B, T, vocab_size]

用 cross entropy 做 next-token prediction。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


torch.manual_seed(42)

# -------------------------
# 1. 数据和 tokenizer
# -------------------------

text = """
the car slows down and the ego vehicle should brake
the vehicle follows the lane and keeps safe distance
the front car stops and the ego car brakes slowly
the road is clear and the vehicle can accelerate
the ego vehicle should avoid collision
the pedestrian crosses the road and the car should stop
the traffic light is red and the ego vehicle waits
the traffic light turns green and the vehicle moves forward
the lane is curved and the vehicle follows the center line
the obstacle is close and the ego vehicle brakes
""" * 100

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


def encode(s: str) -> list[int]:
    return [stoi[ch] for ch in s]


def decode(ids: list[int]) -> str:
    return "".join([itos[i] for i in ids])


data = torch.tensor(encode(text), dtype=torch.long)

# 训练 / 验证划分
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# -------------------------
# 2. 超参数
# -------------------------

batch_size = 32
block_size = 32

max_iters = 1500
eval_interval = 300
eval_iters = 50

learning_rate = 3e-4

n_embd = 64
n_head = 4
n_layer = 2
dropout = 0.1

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# 3. batch 采样
# -------------------------

def get_batch(split: str):
    """
    返回一批训练数据。

    x:
    [B, T]

    y:
    [B, T]

    y 是 x 向右平移一位。
    """
    source = train_data if split == "train" else val_data

    ix = torch.randint(0, len(source) - block_size - 1, (batch_size,))

    x = torch.stack([source[i:i + block_size] for i in ix])
    y = torch.stack([source[i + 1:i + block_size + 1] for i in ix])

    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model):
    """
    简单估计 train / val loss。
    """
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()

        out[split] = losses.mean().item()

    model.train()
    return out


# -------------------------
# 4. 模型组件
# -------------------------

class Head(nn.Module):
    """
    单个 attention head。

    输入:
    x: [B, T, C]

    输出:
    out: [B, T, head_size]
    """

    def __init__(self, head_size: int):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)      # [B, T, H]
        q = self.query(x)    # [B, T, H]
        v = self.value(x)    # [B, T, H]

        H = k.shape[-1]

        # attention score
        # [B, T, H] @ [B, H, T] -> [B, T, T]
        wei = q @ k.transpose(-2, -1) / math.sqrt(H)

        # causal mask
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # attention weight
        wei = F.softmax(wei, dim=-1)

        # dropout
        wei = self.dropout(wei)

        # weighted sum of values
        out = wei @ v  # [B, T, H]

        return out


class MultiHeadAttention(nn.Module):
    """
    多头 attention。

    多个 Head 并行计算，然后 concat 到一起。

    如果：
    n_embd = 64
    n_head = 4

    那么：
    每个 head_size = 16

    4 个 head concat 后又变回 64。
    """

    def __init__(self, num_heads: int, head_size: int):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])

        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 每个 head 输出 [B, T, head_size]
        # concat 后输出 [B, T, n_embd]
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        # 再过一个线性层做混合
        out = self.proj(out)
        out = self.dropout(out)

        return out


class FeedForward(nn.Module):
    """
    前馈网络。

    它对每个 token 位置独立作用。
    不负责 token 之间通信。

    token 之间通信由 attention 完成。
    每个 token 自己的非线性变换由 feed forward 完成。
    """

    def __init__(self, n_embd: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """
    一个 Transformer Block。

    Pre-LN 结构：

    x = x + self_attention(layer_norm(x))
    x = x + feed_forward(layer_norm(x))

    residual connection 的作用：
    让信息和梯度更容易穿过深层网络。
    """

    def __init__(self, n_embd: int, n_head: int):
        super().__init__()

        head_size = n_embd // n_head

        self.sa = MultiHeadAttention(
            num_heads=n_head,
            head_size=head_size
        )

        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # attention 分支
        x = x + self.sa(self.ln1(x))

        # feed forward 分支
        x = x + self.ffwd(self.ln2(x))

        return x


class MiniGPT(nn.Module):
    """
    最小 GPT 语言模型。
    """

    def __init__(self):
        super().__init__()

        # token embedding:
        # [vocab_size, n_embd]
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # position embedding:
        # [block_size, n_embd]
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(n_embd, n_head=n_head)
            for _ in range(n_layer)
        ])

        # final layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        # language modeling head
        # [B, T, n_embd] -> [B, T, vocab_size]
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx:
        [B, T]

        targets:
        [B, T]

        logits:
        [B, T, vocab_size]
        """
        B, T = idx.shape

        if T > block_size:
            raise ValueError(
                f"Sequence length T={T} is larger than block_size={block_size}"
            )

        # token embedding
        tok_emb = self.token_embedding_table(idx)  # [B, T, C]

        # position embedding
        pos = torch.arange(T, device=device)       # [T]
        pos_emb = self.position_embedding_table(pos)  # [T, C]

        # token + position
        x = tok_emb + pos_emb  # [B, T, C]

        # Transformer blocks
        x = self.blocks(x)     # [B, T, C]

        # final layer norm
        x = self.ln_f(x)       # [B, T, C]

        # logits
        logits = self.lm_head(x)  # [B, T, vocab_size]

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(logits_flat, targets_flat)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """
        自回归生成。

        注意：
        因为 position embedding 只有 block_size 长，
        所以每次只把最后 block_size 个 token 喂给模型。
        """
        for _ in range(max_new_tokens):

            # 截断到 block_size
            idx_cond = idx[:, -block_size:]

            logits, loss = self(idx_cond)

            # 只取最后一个位置
            logits = logits[:, -1, :]  # [B, vocab_size]

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(probs, num_samples=1)  # [B, 1]

            idx = torch.cat((idx, idx_next), dim=1)

        return idx


# -------------------------
# 5. 训练
# -------------------------

if __name__ == "__main__":
    print("device:", device)
    print("vocab_size:", vocab_size)
    print("chars:", chars)
    print()

    model = MiniGPT().to(device)

    # 打印参数量
    num_params = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {num_params / 1e6:.3f} M")
    print()

    # 检查 shape
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)

    print("检查 shape:")
    print("xb.shape:", xb.shape)
    print("yb.shape:", yb.shape)
    print("logits.shape:", logits.shape)
    print("loss:", loss.item())
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(max_iters):

        if step % eval_interval == 0:
            losses = estimate_loss(model)
            print(
                f"step {step:4d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

        xb, yb = get_batch("train")

        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print()
    print("生成文本:")

    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=500)[0].tolist()

    print(decode(generated))