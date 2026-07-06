from __future__ import annotations

"""
Day 5 / 02

Instruction tuning: 用不同问法训练同一个 tiny VLM。

和 Day5 / 01 的差别:
    01 只训练 projector，让 image embedding 靠近 label anchors
    02 训练 projector + tiny text backbone + answer head

也就是说，02 不只是“图像分类”。
它让模型在 visual tokens 后面接收不同 prompt，并在 ANSWER token 位置输出答案。
"""

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch
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


def freeze_vision_encoder(model: day4.TinyVLM) -> None:
    for p in model.vision_encoder.parameters():
        p.requires_grad_(False)


def list_trainable_parts(model: day4.TinyVLM) -> list[str]:
    parts = []
    for name, module in [
        ("vision_encoder", model.vision_encoder),
        ("projector", model.projector),
        ("tiny_llm", model.tiny_llm),
        ("answer_head", model.answer_head),
    ]:
        trainable = any(p.requires_grad for p in module.parameters())
        parts.append(f"{name}={'train' if trainable else 'frozen'}")
    return parts


@torch.no_grad()
def show_prompt_predictions(model: day4.TinyVLM, sample: dict, device: str) -> None:
    model.eval()

    print("同一张图，不同 prompt:")
    for prompt in day4.QUESTION_TEMPLATES:
        image = day4.image_to_tensor(day4.render_bev(sample)).unsqueeze(0).to(device)
        ids = day4.encode_prompt(prompt).unsqueeze(0).to(device)
        logits = model(image, ids)
        pred_id = logits.argmax(dim=-1).item()
        conf = logits.softmax(dim=-1)[0, pred_id].item()
        print(f"  {' '.join(prompt):42s} -> {REASONS[pred_id]} ({conf:.2f})")

    model.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = ensure_data(num_samples=args.num_samples, seed=args.seed)
    train_samples, val_samples = split_samples(samples)

    train_loader = DataLoader(
        day4.TinyVLMDataset(train_samples, prompt_mode="random"),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        day4.TinyVLMDataset(val_samples, prompt_mode="random"),
        batch_size=args.batch_size,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = day4.TinyVLM(len(day4.VOCAB)).to(device)

    # 模拟常见 VLM tuning: 视觉 encoder 冻住，桥接层和语言侧继续调。
    freeze_vision_encoder(model)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total, trainable = day4.count_parameters(model)

    print("目标: Stage-2 instruction tuning")
    print("labels:", REASONS)
    print("device:", device)
    print(f"parameters: total={total:,}, trainable={trainable:,}")
    print("parts:", ", ".join(list_trainable_parts(model)))
    print()

    first = next(iter(train_loader))
    with torch.inference_mode():
        model(first[0].to(device), first[1].to(device), trace=True)
    print()

    show_prompt_predictions(model, val_samples[0], device)
    print()

    for epoch in range(1, args.epochs + 1):
        train_loss = day4.train_one_epoch(model, train_loader, optimizer, device)
        val_acc = day4.evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f}")

    print()
    show_prompt_predictions(model, val_samples[0], device)


if __name__ == "__main__":
    main()
