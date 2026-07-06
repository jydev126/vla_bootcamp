from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.driving_toy import build_vocab, encode_tokens, ensure_data, future_to_tensor, split_samples, state_tokens

MAX_LEN = 16


class TokenObsDataset(Dataset):
    def __init__(self, samples, vocab):
        self.samples = samples
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ids, mask = encode_tokens(state_tokens(sample), self.vocab, MAX_LEN)
        return ids, mask, future_to_tensor(sample)


class TokenWaypointModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 128), nn.ReLU(), nn.Linear(128, 12))

    def forward(self, ids, mask, trace: bool = False):
        x = self.embedding(ids)
        hidden, _ = self.gru(x)
        lengths = mask.long().sum(dim=1).clamp(min=1) - 1
        b = torch.arange(ids.shape[0], device=ids.device)
        last = hidden[b, lengths]
        out = self.head(last)
        if trace:
            print("input_ids:", ids.shape)
            print("token_embeddings:", x.shape)
            print("sequence_hidden:", hidden.shape)
            print("last_valid_hidden:", last.shape)
            print("waypoints:", out.shape)
        return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    samples = ensure_data(num_samples=args.num_samples)
    train, _ = split_samples(samples)
    vocab = build_vocab(train)
    loader = DataLoader(TokenObsDataset(train, vocab), batch_size=args.batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TokenWaypointModel(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()
    first = next(iter(loader))
    print("Day6 step2: structured driving state becomes token sequence")
    model(first[0].to(device), first[1].to(device), trace=True)
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        count = 0
        for ids, mask, y in loader:
            loss = loss_fn(model(ids.to(device), mask.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * ids.shape[0]; count += ids.shape[0]
        print(f"epoch {epoch:02d} loss={total / count:.6f}")


if __name__ == "__main__":
    main()
