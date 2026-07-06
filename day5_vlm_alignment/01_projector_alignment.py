from __future__ import annotations

"""
Day 5 / 01

只训练 projector，理解 VLM alignment。

真实 MiniGPT-4 的第一阶段大意是:
    冻结 vision encoder
    冻结 LLM
    训练中间桥接模块，让图像特征对齐到语言空间

这里做一个最小可运行版本:
    frozen image encoder -> trainable projector -> pooled image token
    pooled image token 和 trainable label anchors 做相似度分类

能训练起来，说明 projector / anchors 真的学会把视觉特征挪到目标语义空间。
"""

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.driving_toy import REASONS, ensure_data, split_samples


def load_day4_train():
    path = ROOT / "day4_vlm" / "04_train_tiny_vlm.py"
    spec = importlib.util.spec_from_file_location("tiny_vlm_train", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


day4 = load_day4_train()


class ProjectorAlignment(nn.Module):
    """
    冻结 vision encoder。

    训练 projector + token_pooler + label_anchors。
    label_anchors 是一个 toy 版“语言空间类别原型”。
    """

    def __init__(self):
        super().__init__()

        self.vision_encoder = day4.bridge.TinyViTEncoder()
        self.projector = day4.bridge.VisionProjector()

        self.token_pooler = nn.Linear(96, 1)
        self.label_anchors = nn.Parameter(torch.randn(len(REASONS), 96) * 0.02)

        for p in self.vision_encoder.parameters():
            p.requires_grad_(False)

    def forward(self, image: torch.Tensor, trace: bool = False) -> torch.Tensor:
        with torch.no_grad():
            visual_features, patch_tokens, patches = self.vision_encoder(image)

        visual_tokens = self.projector(visual_features)

        # 学一个很小的池化器: 哪些 patch token 更该被看见。
        # 左转/右转箭头只占图片小区域，mean pooling 容易把它们冲淡。
        weights = self.token_pooler(visual_tokens).softmax(dim=1)
        pooled = (visual_tokens * weights).sum(dim=1)

        # cosine similarity: image embedding 对每个 label anchor 的相似度。
        anchors = F.normalize(self.label_anchors, dim=-1)
        logits = F.normalize(pooled, dim=-1) @ anchors.t() * 10.0

        if trace:
            print("patches:", tuple(patches.shape))
            print("patch_tokens frozen:", tuple(patch_tokens.shape))
            print("visual_features frozen:", tuple(visual_features.shape))
            print("projected_visual_tokens:", tuple(visual_tokens.shape))
            print("pool_weights:", tuple(weights.shape))
            print("pooled_visual:", tuple(pooled.shape))
            print("alignment_logits:", tuple(logits.shape))

        return logits


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0

    for image, _, label in loader:
        pred = model(image.to(device)).argmax(dim=-1)
        label = label.to(device)
        correct += (pred == label).sum().item()
        total += label.numel()

    model.train()
    return correct / max(total, 1)


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    total_loss = 0.0
    total_count = 0

    for image, _, label in loader:
        image = image.to(device)
        label = label.to(device)

        loss = F.cross_entropy(model(image), label)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * image.shape[0]
        total_count += image.shape[0]

    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = ensure_data(num_samples=args.num_samples, seed=args.seed)
    train_samples, val_samples = split_samples(samples)

    train_loader = DataLoader(day4.TinyVLMDataset(train_samples), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(day4.TinyVLMDataset(val_samples), batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ProjectorAlignment().to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    total, trainable = count_parameters(model)

    print("目标: Stage-1 alignment，只训练 projector")
    print("labels:", REASONS)
    print("device:", device)
    print(f"parameters: total={total:,}, trainable={trainable:,}")
    print("trainable modules: projector, token_pooler, label_anchors")
    print()

    first = next(iter(train_loader))
    with torch.inference_mode():
        model(first[0].to(device), trace=True)
    print()

    print(f"before training val_acc={evaluate(model, val_loader, device):.3f}")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d} | alignment_loss={train_loss:.4f} | val_acc={val_acc:.3f}")


if __name__ == "__main__":
    main()
