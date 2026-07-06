"""
Day 1 Project 3: Gradient flow check
Task: print gradient norms and compare normal backward, no_grad, and detach.

Key concepts:
- requires_grad: whether this tensor/parameter participates in gradient calculation.
- loss.backward(): compute gradients and put them into parameter.grad.
- no_grad: do not build computation graph for a block of code.
- detach: cut gradient flow at a specific tensor.
"""

import torch
import torch.nn as nn
import torch.optim as optim


def choose_device() -> torch.device:
    try:
        if torch.cuda.is_available():
            _ = torch.randn(1, device="cuda")
            return torch.device("cuda")
    except Exception as e:
        print(f"[WARN] CUDA is not usable, fallback to CPU. Reason: {e}")
    return torch.device("cpu")


class SplitMLP(nn.Module):
    """Split model into encoder and head so detach effect is easy to see."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, detach_hidden: bool = False) -> torch.Tensor:
        hidden = self.encoder(x)

        if detach_hidden:
            hidden = hidden.detach()

        out = self.head(hidden)
        return out


def make_data(device: torch.device):
    x = torch.linspace(-10, 10, 1000).unsqueeze(1).to(device)
    y = torch.sin(x) + 0.3 * x
    return x, y


def print_grad_norms(model: nn.Module, title: str):
    print(f"\n--- {title} ---")
    for name, p in model.named_parameters():
        if p.grad is None:
            print(f"{name:20s} grad=None")
        else:
            print(f"{name:20s} grad_norm={p.grad.norm().item():.6f}")


def experiment_normal_backward(device: torch.device):
    print("\n================ Experiment A: normal backward ================")
    x, y = make_data(device)
    model = SplitMLP().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    pred = model(x, detach_hidden=False)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    print_grad_norms(model, "after loss.backward()")
    optimizer.step()

    print("Expected: encoder and head both have gradients.")


def experiment_no_grad(device: torch.device):
    print("\n================ Experiment B: torch.no_grad ================")
    x, y = make_data(device)
    model = SplitMLP().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    try:
        with torch.no_grad():
            pred = model(x, detach_hidden=False)

        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
    except RuntimeError as e:
        print("RuntimeError is expected here:")
        print(e)

    print("Expected: no_grad does not build computation graph, so backward cannot work.")


def experiment_detach(device: torch.device):
    print("\n================ Experiment C: hidden.detach ================")
    x, y = make_data(device)
    model = SplitMLP().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    pred = model(x, detach_hidden=True)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    print_grad_norms(model, "after loss.backward() with detach_hidden=True")
    optimizer.step()

    print("Expected: encoder has no gradient; head still has gradients.")


def experiment_freeze_requires_grad(device: torch.device):
    print("\n================ Experiment D: requires_grad=False ================")
    x, y = make_data(device)
    model = SplitMLP().to(device)
    loss_fn = nn.MSELoss()

    # Freeze encoder parameters. This is common in staged training.
    for p in model.encoder.parameters():
        p.requires_grad = False

    # Optimizer receives only trainable parameters.
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    pred = model(x, detach_hidden=False)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    print_grad_norms(model, "after backward with frozen encoder")
    optimizer.step()

    print("Expected: frozen encoder has grad=None; head has gradients.")


def main() -> None:
    torch.manual_seed(42)
    device = choose_device()
    print("device:", device)

    experiment_normal_backward(device)
    experiment_no_grad(device)
    experiment_detach(device)
    experiment_freeze_requires_grad(device)


if __name__ == "__main__":
    main()