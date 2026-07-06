from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.driving_toy import COMMANDS, ensure_data, future_to_tensor, split_samples


def features(sample: dict) -> torch.Tensor:
    values = [sample["ego_speed"] / 15.0, sample["lead_distance"] / 45.0, sample["rel_speed"] / 8.0]
    values += [1.0 if sample["command"] == cmd else 0.0 for cmd in COMMANDS]
    for x, y in sample["history"]:
        values += [x / 20.0, y / 4.0]
    return torch.tensor(values, dtype=torch.float32)


class WaypointDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return features(sample), future_to_tensor(sample)


class MLPWaypointHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(13, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 12))

    def forward(self, x, trace: bool = False):
        y = self.net(x)
        if trace:
            print("features:", x.shape)
            print("waypoints:", y.shape)
        return y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    samples = ensure_data(num_samples=args.num_samples)
    train, val = split_samples(samples)
    loader = DataLoader(WaypointDataset(train), batch_size=args.batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPWaypointHead().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    loss_fn = nn.MSELoss()
    first = next(iter(loader))
    print("Day6 step1: action is numeric waypoints, not text")
    print("source direction = ACT/ALOHA action learning before VLA")
    model(first[0].to(device), trace=True)
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        count = 0
        for x, y in loader:
            loss = loss_fn(model(x.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * x.shape[0]; count += x.shape[0]
        print(f"epoch {epoch:02d} loss={total / count:.6f}")


if __name__ == "__main__":
    main()
