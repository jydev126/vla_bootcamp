"""
Day 1 Project 1: MLP regression
Task: fit y = sin(x) + 0.3 * x

Key concepts:
- Tensor / shape
- nn.Module / forward
- loss / backward
- optimizer.step
- train / eval
- checkpoint
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


OUTPUT_DIR = Path("day1_pytorch/outputs")
CKPT_DIR = Path("day1_pytorch/checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def choose_device() -> torch.device:
    """Use CUDA only if it is really usable; otherwise fall back to CPU."""
    try:
        if torch.cuda.is_available():
            # Some machines have CUDA visible but incompatible drivers.
            _ = torch.randn(1, device="cuda")
            return torch.device("cuda")
    except Exception as e:
        print(f"[WARN] CUDA is not usable, fallback to CPU. Reason: {e}")
    return torch.device("cpu")


class MLP(nn.Module):
    """A small multi-layer perceptron for 1D regression."""

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


def main() -> None:
    torch.manual_seed(42)
    device = choose_device()
    print("device:", device)

    # 1. Generate training data.
    # x: [N, 1], y: [N, 1]
    # N is number of samples; 1 is feature dimension.
    x = torch.linspace(-10, 10, 1000).unsqueeze(1)
    y = torch.sin(x) + 0.3 * x

    x = x.to(device)
    y = y.to(device)

    print("x:", x.shape)
    print("y:", y.shape)

    # 2. Define model, loss, optimizer.
    model = MLP().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Print pred shape once before training.
    with torch.no_grad():
        pred = model(x)
        print("pred:", pred.shape)

    # 3. Train.
    loss_history = []
    model.train()
    num_steps = 2000

    for step in range(num_steps):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        if step % 100 == 0:
            print(f"step={step:04d}, loss={loss.item():.6f}")

    # 4. Save loss curve.
    plt.figure()
    plt.plot(loss_history)
    plt.xlabel("step")
    plt.ylabel("MSE loss")
    plt.title("Training Loss")
    plt.savefig(OUTPUT_DIR / "01_loss_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 5. Save prediction curve.
    model.eval()
    with torch.no_grad():
        pred = model(x)

    x_cpu = x.detach().cpu().squeeze().numpy()
    y_cpu = y.detach().cpu().squeeze().numpy()
    pred_cpu = pred.detach().cpu().squeeze().numpy()

    plt.figure()
    plt.plot(x_cpu, y_cpu, label="true")
    plt.plot(x_cpu, pred_cpu, label="pred")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("MLP Regression: y = sin(x) + 0.3x")
    plt.legend()
    plt.savefig(OUTPUT_DIR / "01_prediction_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 6. Save checkpoint.
    torch.save(
        {
            "step": num_steps,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.item(),
        },
        CKPT_DIR / "01_mlp_regression.pt",
    )

    print("Saved:")
    print(" -", OUTPUT_DIR / "01_loss_curve.png")
    print(" -", OUTPUT_DIR / "01_prediction_curve.png")
    print(" -", CKPT_DIR / "01_mlp_regression.pt")


if __name__ == "__main__":
    main()