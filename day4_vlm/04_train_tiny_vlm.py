from __future__ import annotations

"""
Day 4 / 04

训练一个 tiny VLM，让它看一张 BEV 小图，然后回答规划原因。

输入:
    image: 一张 64x64 的道路俯视图
    text:  "BOS what affects ego planning ANSWER"

输出:
    REASONS 里的一个类别

这不是大模型，但结构和 VLM 主干很像:

    image -> TinyViTEncoder -> visual_features
    visual_features -> VisionProjector -> visual_tokens
    text ids -> token_embedding -> text_tokens
    concat(visual_tokens, text_tokens) -> TinyTextBackbone
    ANSWER 位置 hidden -> answer_head -> logits
"""

import argparse
import importlib.util
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.driving_toy import REASONS, ensure_data, image_to_tensor, reason_from_sample, render_bev, split_samples


def load_bridge():
    path = Path(__file__).with_name("03_minigpt4_bridge.py")
    spec = importlib.util.spec_from_file_location("minigpt4_bridge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


bridge = load_bridge()

QUESTION_TEMPLATES = [
    ["BOS", "what", "affects", "ego", "planning", "ANSWER"],
    ["BOS", "choose", "driving", "reason", "now", "ANSWER"],
    ["BOS", "look", "image", "and", "decide", "ANSWER"],
]

VOCAB_TOKENS = ["PAD"]
for template in QUESTION_TEMPLATES:
    for token in template:
        if token not in VOCAB_TOKENS:
            VOCAB_TOKENS.append(token)

VOCAB = {token: i for i, token in enumerate(VOCAB_TOKENS)}


def encode_prompt(tokens: list[str]) -> torch.Tensor:
    return torch.tensor([VOCAB[token] for token in tokens], dtype=torch.long)


class TinyVLMDataset(Dataset):
    """
    每个样本返回:
        image: [3, 64, 64]
        input_ids: [6]
        label: []

    label 是规划原因，不是原始 command。
    例如 command=keep 但前车太近，也应该归到 brake_due_to_close_lead。
    """

    def __init__(self, samples: list[dict], prompt_mode: str = "fixed"):
        self.samples = samples
        self.prompt_mode = prompt_mode
        self.reason_to_id = {name: i for i, name in enumerate(REASONS)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]

        if self.prompt_mode == "random":
            prompt = random.choice(QUESTION_TEMPLATES)
        else:
            prompt = QUESTION_TEMPLATES[0]

        image = image_to_tensor(render_bev(sample))
        input_ids = encode_prompt(prompt)
        label = torch.tensor(self.reason_to_id[reason_from_sample(sample)], dtype=torch.long)

        return image, input_ids, label


class TinyVLM(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()

        self.vision_encoder = bridge.TinyViTEncoder()
        self.projector = bridge.VisionProjector()
        self.tiny_llm = bridge.TinyTextBackbone(vocab_size)
        self.answer_head = nn.Sequential(
            nn.LayerNorm(96),
            nn.Linear(96, len(REASONS)),
        )

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, trace: bool = False) -> torch.Tensor:
        visual_features, patch_tokens, patches = self.vision_encoder(pixel_values)

        visual_tokens = self.projector(visual_features)
        text_tokens = self.tiny_llm.token_embedding(input_ids)

        vlm_inputs = torch.cat([visual_tokens, text_tokens], dim=1)
        hidden = self.tiny_llm(vlm_inputs)

        # ANSWER token 是文字 prompt 的最后一个 token。
        answer_position = visual_tokens.shape[1] + input_ids.shape[1] - 1
        answer_hidden = hidden[:, answer_position]

        logits = self.answer_head(answer_hidden)

        if trace:
            print("pixel_values:", tuple(pixel_values.shape))
            print("patches:", tuple(patches.shape))
            print("patch_tokens:", tuple(patch_tokens.shape))
            print("visual_features:", tuple(visual_features.shape))
            print("visual_tokens:", tuple(visual_tokens.shape))
            print("text_tokens:", tuple(text_tokens.shape))
            print("vlm_inputs:", tuple(vlm_inputs.shape))
            print("answer_hidden:", tuple(answer_hidden.shape))
            print("logits:", tuple(logits.shape))

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

    for image, ids, label in loader:
        image = image.to(device)
        ids = ids.to(device)
        label = label.to(device)

        pred = model(image, ids).argmax(dim=-1)
        correct += (pred == label).sum().item()
        total += label.numel()

    model.train()
    return correct / max(total, 1)


@torch.no_grad()
def show_predictions(model: nn.Module, samples: list[dict], device: str, title: str) -> None:
    model.eval()
    dataset = TinyVLMDataset(samples, prompt_mode="fixed")

    print(title)
    for i in range(min(5, len(dataset))):
        image, ids, label = dataset[i]
        logits = model(image.unsqueeze(0).to(device), ids.unsqueeze(0).to(device))
        pred_id = logits.argmax(dim=-1).item()
        probs = logits.softmax(dim=-1)[0]

        sample = samples[i]
        print(
            f"  sample {i}: "
            f"command={sample['command']:>5s}, "
            f"lead={sample['lead_distance']:>5.1f}, "
            f"rel={sample['rel_speed']:>5.1f}, "
            f"target={REASONS[label.item()]}, "
            f"pred={REASONS[pred_id]}, "
            f"conf={probs[pred_id].item():.2f}"
        )
    model.train()


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_count = 0

    for image, ids, label in loader:
        image = image.to(device)
        ids = ids.to(device)
        label = label.to(device)

        logits = model(image, ids)
        loss = loss_fn(logits, label)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * image.shape[0]
        total_count += image.shape[0]

    return total_loss / max(total_count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    samples = ensure_data(num_samples=args.num_samples, seed=args.seed)
    train_samples, val_samples = split_samples(samples)

    train_loader = DataLoader(
        TinyVLMDataset(train_samples, prompt_mode="fixed"),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(TinyVLMDataset(val_samples, prompt_mode="fixed"), batch_size=args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TinyVLM(len(VOCAB)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total, trainable = count_parameters(model)

    print("目标: 训练一个 tiny VLM 做图像问答分类")
    print("labels:", REASONS)
    print("prompt:", " ".join(QUESTION_TEMPLATES[0]))
    print("device:", device)
    print(f"parameters: total={total:,}, trainable={trainable:,}")
    print()

    first = next(iter(train_loader))
    with torch.inference_mode():
        model(first[0].to(device), first[1].to(device), trace=True)
    print()

    show_predictions(model, val_samples, device, title="训练前预测:")
    print()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch:02d} | train_loss={train_loss:.4f} | val_acc={val_acc:.3f}")

    print()
    show_predictions(model, val_samples, device, title="训练后预测:")


if __name__ == "__main__":
    main()
