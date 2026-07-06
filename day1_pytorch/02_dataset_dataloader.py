"""
Day 1 Project 2: Custom Dataset and DataLoader
Task: wrap y = sin(x) + 0.3 * x data into Dataset and train with mini-batches.

Key concepts:
- Dataset returns one sample.
- DataLoader groups samples into a batch.
- Batch first dimension is B.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("day1_pytorch/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def choose_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            _ = torch.randn(1, device="cuda")
            return torch.device("cuda")
    except Exception as e:
        print(f"[WARN] CUDA is not usable, fallback to CPU. Reason: {e}")
    return torch.device("cpu")


class SineDataset(Dataset):
    """A dataset that returns one (x, y) pair each time."""

    def __init__(self, n_samples: int = 1000):
        self.x = torch.linspace(-10, 10, n_samples).unsqueeze(1)
        self.y = torch.sin(self.x) + 0.3 * self.x

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        # Return one sample.
        # x[idx].shape is [1], y[idx].shape is [1].
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one_setting(batch_size: int, device: torch.device):
    print(f"\n==== batch_size = {batch_size} ====")

    dataset = SineDataset(n_samples=1000)
    print("single sample x shape:", dataset[0][0].shape)
    print("single sample y shape:", dataset[0][1].shape)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MLP().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    loss_history = []
    num_epochs = 20

    for epoch in range(num_epochs):
        model.train()

        for batch_idx, (x_batch, y_batch) in enumerate(loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_history.append(loss.item())

            if epoch == 0 and batch_idx == 0:
                print("x_batch:", x_batch.shape)
                print("y_batch:", y_batch.shape)
                print("pred:", pred.shape)

        print(f"epoch={epoch:02d}, last_loss={loss.item():.6f}")

    return loss_history


def main() -> None:
    torch.manual_seed(42)
    device = choose_device()
    print("device:", device)

    histories = {}
    for batch_size in [4, 32, 256]:
        histories[batch_size] = train_one_setting(batch_size, device)

    plt.figure()
    for batch_size, loss_history in histories.items():
        plt.plot(loss_history, label=f"batch_size={batch_size}")
    plt.xlabel("optimization step")
    plt.ylabel("MSE loss")
    plt.title("Loss Curves with Different Batch Sizes")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "02_batch_size_loss_compare.png", dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved:")
    print(" -", OUTPUT_DIR / "02_batch_size_loss_compare.png")


if __name__ == "__main__":
    main()