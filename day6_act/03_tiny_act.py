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
NUM_WAYPOINTS = 6


class ACTDataset(Dataset):
    def __init__(self, samples, vocab):
        self.samples = samples
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ids, mask = encode_tokens(state_tokens(sample), self.vocab, MAX_LEN)
        return ids, mask, future_to_tensor(sample).view(NUM_WAYPOINTS, 2)


class TinyACT(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, hidden_dim)
        enc = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 4, batch_first=True, activation="gelu")
        dec = nn.TransformerDecoderLayer(hidden_dim, 4, hidden_dim * 4, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.decoder = nn.TransformerDecoder(dec, num_layers=2)
        self.action_queries = nn.Parameter(torch.randn(1, NUM_WAYPOINTS, hidden_dim) * 0.02)
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, ids, mask, trace: bool = False):
        bsz, seq_len = ids.shape
        pos = torch.arange(seq_len, device=ids.device).unsqueeze(0).expand(bsz, seq_len)
        obs = self.embedding(ids) + self.pos(pos)
        memory = self.encoder(obs, src_key_padding_mask=~mask.bool())
        queries = self.action_queries.expand(bsz, -1, -1)
        action_hidden = self.decoder(queries, memory, memory_key_padding_mask=~mask.bool())
        actions = self.head(action_hidden)
        if trace:
            print("input_ids:", ids.shape)
            print("observation_memory:", memory.shape)
            print("action_queries:", queries.shape)
            print("action_hidden:", action_hidden.shape)
            print("action_chunk:", actions.shape)
        return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    samples = ensure_data(num_samples=args.num_samples)
    train, _ = split_samples(samples)
    vocab = build_vocab(train)
    loader = DataLoader(ACTDataset(train, vocab), batch_size=args.batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyACT(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()
    first = next(iter(loader))
    print("source = ACT: https://github.com/tonyzhaozh/act")
    model(first[0].to(device), first[1].to(device), trace=True)
    print("target action_chunk:", first[2].shape)
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
