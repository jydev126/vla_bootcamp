from __future__ import annotations

"""
Day 5 / 02

把 01 的 caption VLM 改成 driving-scene visual instruction tuning。

关键变化:
    普通 VLM:
        image -> caption

    driving VLM:
        image + driving question -> short grounded answer

这里仍然使用真实预训练组件:
    frozen CLIP vision encoder
    frozen DistilGPT2 language model
    trainable prefix projector

默认数据仍来自公开 Flickr30k，但会筛选包含 road / street / car / bus / person
等词的样本，并从公开 caption 派生轻量驾驶 QA。它不是 DriveLM 替代品，
而是一个能在普通电脑上跑起来的“先吃透 VLM 训练形态”的桥。
"""

import argparse
import importlib.util
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = Path(__file__).with_name("01_clip_prefix_captioner.py")
DEFAULT_SAVE_PATH = "common/outputs/day5_real_vlm/driving_qa_projector.pt"


def load_core():
    spec = importlib.util.spec_from_file_location("day5_clip_prefix_captioner", CORE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


core = load_core()

DRIVING_TERMS = {
    "road",
    "street",
    "traffic",
    "intersection",
    "lane",
    "sidewalk",
    "crosswalk",
    "car",
    "cars",
    "vehicle",
    "vehicles",
    "truck",
    "trucks",
    "bus",
    "buses",
    "taxi",
    "motorcycle",
    "bicycle",
    "bike",
    "cyclist",
    "pedestrian",
    "person",
    "people",
    "man",
    "woman",
    "child",
}

VEHICLE_TERMS = {"car", "cars", "vehicle", "vehicles", "truck", "trucks", "bus", "buses", "taxi", "van"}
PERSON_TERMS = {"person", "people", "man", "woman", "child", "children", "pedestrian", "pedestrians"}
CYCLIST_TERMS = {"bicycle", "bike", "bicyclist", "cyclist", "motorcycle", "motorcyclist"}
ROAD_TERMS = {"road", "street", "traffic", "intersection", "lane", "sidewalk", "crosswalk"}


def caption_blob(row: dict[str, Any]) -> str:
    captions = row["caption"]
    if isinstance(captions, str):
        return captions.lower()
    return " ".join(captions).lower()


def has_any(text: str, terms: set[str]) -> bool:
    padded = f" {text.replace('.', ' ').replace(',', ' ')} "
    return any(f" {term} " in padded for term in terms)


def is_driving_row(row: dict[str, Any]) -> bool:
    text = caption_blob(row)
    return has_any(text, DRIVING_TERMS)


def load_driving_rows(
    dataset_name: str,
    target_split: str,
    max_samples: int,
    seed: int,
    streaming: bool,
) -> list[dict[str, Any]]:
    load_dataset, *_ = core.require_hf_libraries()
    dataset = load_dataset(dataset_name, split="test", streaming=streaming)

    if streaming:
        rows = []
        for row in dataset:
            if row.get("split") != target_split:
                continue
            if not is_driving_row(row):
                continue
            rows.append(row)
            if len(rows) >= max_samples:
                break
        return rows

    dataset = dataset.filter(lambda row: row["split"] == target_split and is_driving_row(row))
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    return [dataset[i] for i in range(len(dataset))]


def scene_tags(row: dict[str, Any]) -> dict[str, bool]:
    text = caption_blob(row)
    return {
        "vehicle": has_any(text, VEHICLE_TERMS),
        "person": has_any(text, PERSON_TERMS),
        "cyclist": has_any(text, CYCLIST_TERMS),
        "road": has_any(text, ROAD_TERMS),
    }


def road_user_answer(tags: dict[str, bool]) -> str:
    users = []
    if tags["vehicle"]:
        users.append("vehicles")
    if tags["person"]:
        users.append("pedestrians")
    if tags["cyclist"]:
        users.append("cyclists")
    if not users:
        return "no clear road users"
    if len(users) == 1:
        return users[0]
    return ", ".join(users[:-1]) + " and " + users[-1]


def risk_answer(tags: dict[str, bool]) -> str:
    if tags["person"] and tags["road"]:
        return "people near the roadway"
    if tags["cyclist"] and tags["road"]:
        return "a cyclist sharing the road"
    if tags["vehicle"] and tags["road"]:
        return "nearby vehicles in traffic"
    if tags["person"]:
        return "nearby people"
    if tags["vehicle"]:
        return "nearby vehicles"
    return "no obvious traffic risk"


def action_answer(tags: dict[str, bool]) -> str:
    if tags["person"] and tags["road"]:
        return "slow down and watch for pedestrians"
    if tags["cyclist"] and tags["road"]:
        return "slow down and give the cyclist space"
    if tags["vehicle"] and tags["road"]:
        return "keep a safe distance from vehicles"
    return "continue carefully"


def qa_pairs_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    tags = scene_tags(row)
    return [
        {
            "question": "What road users are visible?",
            "answer": road_user_answer(tags),
            "kind": "perception",
        },
        {
            "question": "What should the driving assistant pay attention to?",
            "answer": risk_answer(tags),
            "kind": "risk",
        },
        {
            "question": "What is a reasonable high level driving response?",
            "answer": action_answer(tags),
            "kind": "planning",
        },
    ]


class DrivingQADataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], questions_per_image: int = 3):
        self.items = []
        for row in rows:
            for pair in qa_pairs_from_row(row)[:questions_per_image]:
                self.items.append(
                    {
                        "image": row["image"].convert("RGB"),
                        "question": pair["question"],
                        "answer": pair["answer"],
                        "kind": pair["kind"],
                        "filename": row.get("filename", str(len(self.items))),
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


@dataclass
class DrivingQACollator:
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
            prompt = f"Question: {example['question']}\nAnswer:"
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
            "kinds": [example["kind"] for example in examples],
            "filenames": [example["filename"] for example in examples],
        }


@torch.no_grad()
def generate_answer(
    model: core.ClipPrefixCaptioner,
    pixel_values: torch.Tensor,
    tokenizer: Any,
    question: str,
    device: str,
    max_new_tokens: int,
) -> str:
    model.eval()
    pixel_values = pixel_values.to(device)
    prefix_embeds = model.encode_image_prefix(pixel_values)
    prompt_ids = tokenizer(f"Question: {question}\nAnswer:", add_special_tokens=False, return_tensors="pt").input_ids.to(device)
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
def show_qa_generations(
    title: str,
    model: core.ClipPrefixCaptioner,
    dataset: DrivingQADataset,
    collator: DrivingQACollator,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
    count: int,
) -> None:
    print(title)
    model.eval()
    for idx in range(min(count, len(dataset))):
        example = dataset[idx]
        batch = core.move_batch(collator([example]), device)
        generated = generate_answer(model, batch["pixel_values"], tokenizer, example["question"], device, max_new_tokens)
        print(f"  image={example['filename']} kind={example['kind']}")
        print(f"    question: {example['question']}")
        print(f"    target:   {example['answer']}")
        print(f"    pred:     {generated}")


@torch.no_grad()
def planning_rank_probe(
    model: core.ClipPrefixCaptioner,
    rows: list[dict[str, Any]],
    collator: DrivingQACollator,
    device: str,
    max_items: int,
) -> tuple[int, int]:
    model.eval()
    correct = 0
    total = 0
    candidates = [
        "continue carefully",
        "keep a safe distance from vehicles",
        "slow down and watch for pedestrians",
        "slow down and give the cyclist space",
    ]

    for row in rows[:max_items]:
        tags = scene_tags(row)
        target = action_answer(tags)
        if target not in candidates:
            continue

        examples = []
        for candidate in candidates:
            examples.append(
                {
                    "image": row["image"].convert("RGB"),
                    "question": "What is a reasonable high level driving response?",
                    "answer": candidate,
                    "kind": "planning",
                    "filename": row.get("filename", str(total)),
                }
            )

        batch = core.move_batch(collator(examples), device)
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        nll = core.token_nll_from_logits(outputs.logits, batch["labels"])
        pred = candidates[nll.argmin().item()]
        correct += int(pred == target)
        total += 1

    return correct, total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=core.DEFAULT_DATASET)
    parser.add_argument("--clip-model", default=core.DEFAULT_CLIP)
    parser.add_argument("--llm-model", default=core.DEFAULT_LLM)
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--val-samples", type=int, default=48)
    parser.add_argument("--questions-per-image", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--projector-hidden", type=int, default=1024)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--rank-probe-size", type=int, default=12)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--save-path", default=DEFAULT_SAVE_PATH)
    parser.add_argument("--load-path", default="")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device or core.get_default_device()

    print("目标: driving-scene visual instruction tuning")
    print("普通 caption 只问图里有什么；driving QA 要问 road users / risk / planning response")
    print(f"dataset: {args.dataset} filtered by driving terms")
    print(f"clip: {args.clip_model}")
    print(f"llm: {args.llm_model}")
    print(f"device: {device}")
    print()

    train_rows = load_driving_rows(args.dataset, "train", args.train_samples, args.seed, args.streaming)
    val_rows = load_driving_rows(args.dataset, "test", args.val_samples, args.seed + 1, args.streaming)
    if not train_rows or not val_rows:
        raise SystemExit("没有筛到 driving rows。可以增大 --train-samples/--val-samples 或去掉 --streaming 后重试。")

    model, clip_processor, tokenizer = core.build_model_and_processors(args, device)
    core.load_trainable_checkpoint(model, args.load_path)
    collator = DrivingQACollator(
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        prefix_length=args.prefix_length,
        max_answer_tokens=args.max_answer_tokens,
    )

    train_dataset = DrivingQADataset(train_rows, questions_per_image=args.questions_per_image)
    val_dataset = DrivingQADataset(val_rows, questions_per_image=args.questions_per_image)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=not args.streaming, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    total, trainable = core.count_parameters(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    print(f"driving rows: train={len(train_rows)}, val={len(val_rows)}")
    print(f"qa examples: train={len(train_dataset)}, val={len(val_dataset)}")
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

    show_qa_generations("训练前 driving QA generation:", model, val_dataset, collator, tokenizer, device, args.max_new_tokens, 4)
    before_eval = core.evaluate_loss(model, val_loader, device, args.eval_batches)
    before_rank = planning_rank_probe(model, val_rows, collator, device, args.rank_probe_size)
    print(f"训练前 eval: loss={before_eval['loss']:.4f} ppl={before_eval['ppl']:.2f}")
    print(f"训练前 planning-rank probe: {before_rank[0]}/{before_rank[1]}")
    print()

    for epoch in range(1, args.epochs + 1):
        train_loss = core.train_one_epoch(model, train_loader, optimizer, device, args.log_every)
        val_eval = core.evaluate_loss(model, val_loader, device, args.eval_batches)
        rank = planning_rank_probe(model, val_rows, collator, device, args.rank_probe_size)
        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_eval['loss']:.4f} | val_ppl={val_eval['ppl']:.2f} | "
            f"planning_rank={rank[0]}/{rank[1]}"
        )

    print()
    show_qa_generations("训练后 driving QA generation:", model, val_dataset, collator, tokenizer, device, args.max_new_tokens, 4)
    core.save_trainable_checkpoint(model, args.save_path, args)


if __name__ == "__main__":
    main()
