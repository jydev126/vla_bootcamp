from __future__ import annotations

"""
Day 4 / 05

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
    stats = evaluate_detailed(model, loader, device)
    return stats["accuracy"]


@torch.no_grad()
def evaluate_detailed(model: nn.Module, loader: DataLoader, device: str) -> dict:
    was_training = model.training
    model.eval()
    confusion = torch.zeros(len(REASONS), len(REASONS), dtype=torch.long)
    examples = []

    for image, _, label in loader:
        image = image.to(device)
        label = label.to(device)
        logits = model(image)
        pred = logits.argmax(dim=-1)
        probs = logits.softmax(dim=-1)

        for target_id, pred_id in zip(label.cpu(), pred.cpu()):
            confusion[target_id, pred_id] += 1

        if len(examples) < 5:
            batch_examples = min(image.shape[0], 5 - len(examples))
            for i in range(batch_examples):
                pred_id = pred[i].item()
                examples.append(
                    {
                        "target": REASONS[label[i].item()],
                        "pred": REASONS[pred_id],
                        "conf": probs[i, pred_id].item(),
                    }
                )

    total = confusion.sum().item()
    correct = confusion.diag().sum().item()
    if was_training:
        model.train()
    return {
        "accuracy": correct / max(total, 1),
        "correct": correct,
        "total": total,
        "confusion": confusion,
        "examples": examples,
    }


def print_eval_report(title: str, stats: dict) -> None:
    confusion = stats["confusion"]
    print(title)
    print(f"  accuracy={stats['accuracy']:.3f} ({stats['correct']}/{stats['total']})")
    print("  per-class:")
    for i, name in enumerate(REASONS):
        row_total = confusion[i].sum().item()
        row_correct = confusion[i, i].item()
        acc = row_correct / max(row_total, 1)
        print(f"    {name:26s} {acc:.3f} ({row_correct}/{row_total})")
    print("  confusion matrix rows=target cols=pred:")
    print("    " + " ".join(f"{i:>4d}" for i in range(len(REASONS))))
    for i, name in enumerate(REASONS):
        values = " ".join(f"{v.item():>4d}" for v in confusion[i])
        print(f"    {i} {name:24s} {values}")
    print("  examples:")
    for example in stats["examples"]:
        print(f"    target={example['target']:26s} pred={example['pred']:26s} conf={example['conf']:.2f}")


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

    print_eval_report("训练前 eval:", evaluate_detailed(model, val_loader, device))
    print()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d} | alignment_loss={train_loss:.4f} | val_acc={val_acc:.3f}")

    print()
    print_eval_report("训练后 eval:", evaluate_detailed(model, val_loader, device))


if __name__ == "__main__":
    main()
