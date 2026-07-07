from __future__ import annotations

"""
Day 5 / 04

真实驾驶数据入口版。

02/03 是教学版: 用公开 Flickr30k 的驾驶相关图片派生轻量 QA，方便小机器跑通。
04 是真实版: 面向 DriveLM / NuScenes-QA / LingoQA / BDD-X 等公开驾驶数据。

本脚本不假装自动下载这些大数据集。真实数据通常有许可、原始图片目录、
多相机文件路径和不同 annotation schema。这里约定一个最小 JSONL 接口:

    {
      "image": "CAM_FRONT/sample.jpg",
      "question": "What should the ego vehicle pay attention to?",
      "answer": "a pedestrian crossing ahead",
      "task": "planning"
    }

多视角也可以:

    {
      "images": {
        "front": "CAM_FRONT/xxx.jpg",
        "front_left": "CAM_FRONT_LEFT/xxx.jpg",
        "front_right": "CAM_FRONT_RIGHT/xxx.jpg"
      },
      "question": "...",
      "answer": "...",
      "candidates": ["continue carefully", "slow down", "stop"],
      "task": "planning"
    }

模型仍然是经典轻量 VLM 路线:
    frozen CLIP vision encoder -> trainable prefix projector -> frozen DistilGPT2
"""

import argparse
import importlib.util
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

CORE_PATH = Path(__file__).with_name("01_clip_prefix_captioner.py")
DEFAULT_SAVE_PATH = "common/outputs/day5_real_vlm/real_driving_vqa_projector.pt"


def load_core():
    spec = importlib.util.spec_from_file_location("day5_clip_prefix_captioner", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_core()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resolve_path(path: str, image_root: Path, jsonl_dir: Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    root_candidate = image_root / raw
    if root_candidate.exists():
        return root_candidate
    return jsonl_dir / raw


def image_paths_from_row(row: dict[str, Any]) -> list[str]:
    if "image" in row:
        return [row["image"]]
    if "images" not in row:
        raise KeyError("每行必须包含 image 或 images 字段")

    images = row["images"]
    if isinstance(images, dict):
        return [images[key] for key in sorted(images)]
    if isinstance(images, list):
        return images
    raise TypeError("images 必须是 list 或 dict")


def make_montage(images: list[Image.Image], view_size: int) -> Image.Image:
    if len(images) == 1:
        return images[0].convert("RGB")

    resized = []
    for image in images:
        image = ImageOps.contain(image.convert("RGB"), (view_size, view_size))
        canvas = Image.new("RGB", (view_size, view_size), color=(20, 22, 24))
        x = (view_size - image.width) // 2
        y = (view_size - image.height) // 2
        canvas.paste(image, (x, y))
        resized.append(canvas)

    cols = min(3, len(resized))
    rows = math.ceil(len(resized) / cols)
    montage = Image.new("RGB", (cols * view_size, rows * view_size), color=(20, 22, 24))
    for idx, image in enumerate(resized):
        x = (idx % cols) * view_size
        y = (idx // cols) * view_size
        montage.paste(image, (x, y))
    return montage


class RealDrivingVQADataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        image_root: str,
        max_samples: int,
        seed: int,
        view_size: int,
    ):
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_dir = self.jsonl_path.resolve().parent
        self.image_root = Path(image_root) if image_root else self.jsonl_dir
        self.view_size = view_size

        rows = read_jsonl(self.jsonl_path)
        rows = [row for row in rows if row.get("question") and row.get("answer")]
        rng = random.Random(seed)
        rng.shuffle(rows)
        if max_samples > 0:
            rows = rows[:max_samples]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        paths = [resolve_path(path, self.image_root, self.jsonl_dir) for path in image_paths_from_row(row)]
        images = []
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"image not found: {path}")
            images.append(Image.open(path).convert("RGB"))

        return {
            "image": make_montage(images, self.view_size),
            "question": row["question"].strip(),
            "answer": row["answer"].strip(),
            "candidates": row.get("candidates", []),
            "task": row.get("task", "unknown"),
            "scene_id": row.get("scene_id", row.get("id", str(idx))),
        }


@dataclass
class RealDrivingVQACollator:
    clip_processor: Any
    tokenizer: Any
    prefix_length: int
    max_answer_tokens: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = [example["image"] for example in examples]
        pixel_values = self.clip_processor(images=images, return_tensors="pt").pixel_values

        text_ids_list: list[list[int]] = []
        text_mask_list: list[list[int]] = []
        label_list: list[list[int]] = []

        for example in examples:
            task = example.get("task", "driving")
            prompt = f"Task: {task}\nQuestion: {example['question']}\nAnswer:"
            prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
            answer_ids = self.tokenizer(" " + example["answer"], add_special_tokens=False).input_ids
            answer_ids = answer_ids[: max(self.max_answer_tokens - 1, 1)]
            answer_ids.append(self.tokenizer.eos_token_id)

            text_ids = prompt_ids + answer_ids
            labels = [-100] * (self.prefix_length + len(prompt_ids)) + answer_ids

            text_ids_list.append(text_ids)
            text_mask_list.append([1] * len(text_ids))
            label_list.append(labels)

        max_text_len = max(len(ids) for ids in text_ids_list)
        max_label_len = self.prefix_length + max_text_len

        for ids, mask, labels in zip(text_ids_list, text_mask_list, label_list):
            ids.extend([self.tokenizer.pad_token_id] * (max_text_len - len(ids)))
            mask.extend([0] * (max_text_len - len(mask)))
            labels.extend([-100] * (max_label_len - len(labels)))

        prefix_attention = torch.ones(len(examples), self.prefix_length, dtype=torch.long)
        text_attention = torch.tensor(text_mask_list, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(text_ids_list, dtype=torch.long),
            "attention_mask": torch.cat([prefix_attention, text_attention], dim=1),
            "labels": torch.tensor(label_list, dtype=torch.long),
            "questions": [example["question"] for example in examples],
            "answers": [example["answer"] for example in examples],
            "tasks": [example["task"] for example in examples],
            "scene_ids": [example["scene_id"] for example in examples],
        }


@torch.no_grad()
def generate_answer(
    model: core.ClipPrefixCaptioner,
    pixel_values: torch.Tensor,
    tokenizer: Any,
    task: str,
    question: str,
    device: str,
    max_new_tokens: int,
) -> str:
    model.eval()
    prefix_embeds = model.encode_image_prefix(pixel_values.to(device))
    prompt = f"Task: {task}\nQuestion: {question}\nAnswer:"
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    generated: list[int] = []

    for _ in range(max_new_tokens):
        if generated:
            generated_ids = torch.tensor([generated], dtype=torch.long, device=device)
            text_ids = torch.cat([prompt_ids, generated_ids], dim=1)
        else:
            text_ids = prompt_ids

        token_embeds = model.language_model.get_input_embeddings()(text_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
        attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)
        logits = model.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        next_id = logits[:, -1].argmax(dim=-1).item()
        if next_id == tokenizer.eos_token_id:
            break
        generated.append(next_id)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def show_generations(
    title: str,
    model: core.ClipPrefixCaptioner,
    dataset: RealDrivingVQADataset,
    collator: RealDrivingVQACollator,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
    count: int,
) -> None:
    print(title)
    for idx in range(min(count, len(dataset))):
        example = dataset[idx]
        batch = core.move_batch(collator([example]), device)
        pred = generate_answer(
            model=model,
            pixel_values=batch["pixel_values"],
            tokenizer=tokenizer,
            task=example["task"],
            question=example["question"],
            device=device,
            max_new_tokens=max_new_tokens,
        )
        print(f"  scene={example['scene_id']} task={example['task']}")
        print(f"    question: {example['question']}")
        print(f"    target:   {example['answer']}")
        print(f"    pred:     {pred}")


@torch.no_grad()
def candidate_rank_probe(
    model: core.ClipPrefixCaptioner,
    dataset: RealDrivingVQADataset,
    collator: RealDrivingVQACollator,
    device: str,
    max_items: int,
) -> tuple[int, int]:
    correct = 0
    total = 0

    for idx in range(min(max_items, len(dataset))):
        example = dataset[idx]
        candidates = example.get("candidates") or []
        if not candidates or example["answer"] not in candidates:
            continue

        candidate_examples = []
        for candidate in candidates:
            candidate_examples.append({**example, "answer": candidate})

        batch = core.move_batch(collator(candidate_examples), device)
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        nll = core.token_nll_from_logits(outputs.logits, batch["labels"])
        pred = candidates[nll.argmin().item()]
        correct += int(pred == example["answer"])
        total += 1

    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--val-jsonl", required=True)
    parser.add_argument("--image-root", default="")
    parser.add_argument("--clip-model", default=core.DEFAULT_CLIP)
    parser.add_argument("--llm-model", default=core.DEFAULT_LLM)
    parser.add_argument("--train-samples", type=int, default=2000)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--projector-hidden", type=int, default=1024)
    parser.add_argument("--max-answer-tokens", type=int, default=48)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--view-size", type=int, default=224)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--rank-probe-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH)
    parser.add_argument("--load-path", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = args.device or core.get_default_device()

    print("目标: real driving VQA tuning from public driving JSONL")
    print("数据层应来自 DriveLM / NuScenes-QA / LingoQA / BDD-X 等公开驾驶数据的本地预处理结果")
    print(f"train_jsonl: {args.train_jsonl}")
    print(f"val_jsonl: {args.val_jsonl}")
    print(f"image_root: {args.image_root or '(jsonl directory)'}")
    print(f"device: {device}")
    print()

    train_dataset = RealDrivingVQADataset(
        jsonl_path=args.train_jsonl,
        image_root=args.image_root,
        max_samples=args.train_samples,
        seed=args.seed,
        view_size=args.view_size,
    )
    val_dataset = RealDrivingVQADataset(
        jsonl_path=args.val_jsonl,
        image_root=args.image_root,
        max_samples=args.val_samples,
        seed=args.seed + 1,
        view_size=args.view_size,
    )

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise SystemExit("train/val JSONL 没有可用样本。请检查 question/answer/image 字段。")

    model, clip_processor, tokenizer = core.build_model_and_processors(args, device)
    core.load_trainable_checkpoint(model, args.load_path)
    collator = RealDrivingVQACollator(
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        prefix_length=args.prefix_length,
        max_answer_tokens=args.max_answer_tokens,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    total, trainable = core.count_parameters(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    print(f"samples: train={len(train_dataset)}, val={len(val_dataset)}")
    print(f"parameters: total={total:,}, trainable={trainable:,}")
    print("trainable modules: prefix_projector, prefix_norm")
    print()

    first = core.move_batch(next(iter(train_loader)), device)
    with torch.inference_mode():
        model(
            pixel_values=first["pixel_values"],
            input_ids=first["input_ids"],
            attention_mask=first["attention_mask"],
            labels=first["labels"],
            trace=True,
        )
    print()

    show_generations("训练前 real driving generation:", model, val_dataset, collator, tokenizer, device, args.max_new_tokens, 3)
    before_eval = core.evaluate_loss(model, val_loader, device, args.eval_batches)
    before_rank = candidate_rank_probe(model, val_dataset, collator, device, args.rank_probe_size)
    print(f"训练前 eval: loss={before_eval['loss']:.4f} ppl={before_eval['ppl']:.2f}")
    if before_rank[1] > 0:
        print(f"训练前 candidate-rank probe: {before_rank[0]}/{before_rank[1]}")
    print()

    for epoch in range(1, args.epochs + 1):
        train_loss = core.train_one_epoch(model, train_loader, optimizer, device, args.log_every)
        val_eval = core.evaluate_loss(model, val_loader, device, args.eval_batches)
        rank = candidate_rank_probe(model, val_dataset, collator, device, args.rank_probe_size)
        suffix = f" | candidate_rank={rank[0]}/{rank[1]}" if rank[1] > 0 else ""
        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_eval['loss']:.4f} | val_ppl={val_eval['ppl']:.2f}{suffix}"
        )

    print()
    show_generations("训练后 real driving generation:", model, val_dataset, collator, tokenizer, device, args.max_new_tokens, 3)
    core.save_trainable_checkpoint(model, args.save_path, args)


if __name__ == "__main__":
    main()
