from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.driving_toy import build_vocab, encode_tokens, ensure_data, future_to_tensor, split_samples, state_tokens

CONTEXT_LEN = 16
ACTION_DIM = 12


def continuous_to_action_ids(target: torch.Tensor, bins: int) -> torch.Tensor:
    return torch.round((target.clamp(-1, 1) + 1.0) * 0.5 * (bins - 1)).long().clamp(0, bins - 1)


class ActionTokenDataset(Dataset):
    def __init__(self, samples, vocab, bins):
        self.samples = samples
        self.vocab = vocab
        self.bins = bins

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ids, mask = encode_tokens(state_tokens(sample), self.vocab, CONTEXT_LEN)
        action_ids = continuous_to_action_ids(future_to_tensor(sample), self.bins)
        return ids, mask, action_ids


class TinyActionLM(nn.Module):
    def __init__(self, vocab_size, bins, hidden_dim=96):
        super().__init__()
        self.bins = bins
        self.context_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(bins + 1, hidden_dim)
        self.pos = nn.Embedding(CONTEXT_LEN + ACTION_DIM, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 4, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.head = nn.Linear(hidden_dim, bins)

    def forward(self, context_ids, action_ids, trace=False):
        bsz = context_ids.shape[0]
        bos = torch.full((bsz, 1), self.bins, dtype=torch.long, device=context_ids.device)
        action_in = torch.cat([bos, action_ids[:, :-1]], dim=1)
        tokens = torch.cat([self.context_embedding(context_ids), self.action_embedding(action_in)], dim=1)
        pos = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0).expand(bsz, -1)
        mask = torch.triu(torch.ones(tokens.shape[1], tokens.shape[1], device=tokens.device), diagonal=1).bool()
        hidden = self.transformer(tokens + self.pos(pos), mask=mask)
        logits = self.head(hidden[:, -ACTION_DIM:])
        if trace:
            print("context_ids:", context_ids.shape)
            print("action_input_ids:", action_in.shape)
            print("full_sequence:", tokens.shape)
            print("action_logits:", logits.shape)
        return logits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    samples = ensure_data(num_samples=args.num_samples)
    train, _ = split_samples(samples)
    vocab = build_vocab(train)
    loader = DataLoader(ActionTokenDataset(train, vocab, args.bins), batch_size=args.batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyActionLM(len(vocab), args.bins).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    first = next(iter(loader))
    print("source = OpenVLA: https://github.com/openvla/openvla")
    model(first[0].to(device), first[2].to(device), trace=True)
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        count = 0
        for ctx, _, act in loader:
            logits = model(ctx.to(device), act.to(device))
            loss = F.cross_entropy(logits.reshape(-1, args.bins), act.to(device).reshape(-1))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * ctx.shape[0]; count += ctx.shape[0]
        print(f"epoch {epoch:02d} ce={total / count:.4f}")


if __name__ == "__main__":
    main()
