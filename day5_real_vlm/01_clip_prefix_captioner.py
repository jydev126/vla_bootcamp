from __future__ import annotations

"""
Day 5 / 01

用真实预训练组件搭一个小型可训练 VLM:

    image -> frozen CLIP vision encoder -> trainable prefix projector
    prompt/caption -> frozen DistilGPT2 token embeddings
    concat([visual_prefix, text_tokens]) -> frozen DistilGPT2 -> caption loss

这是一条经典路线的轻量实现:
冻结视觉塔和 LLM，只训练中间桥接层，让图像变成 LLM 能接住的 prefix tokens。
它和 ClipCap / BLIP-2 / MiniGPT-4 都属于“视觉特征接入语言模型”的家族。
"""

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET = "nlphuji/flickr30k"
DEFAULT_CLIP = "openai/clip-vit-base-patch32"
DEFAULT_LLM = "distilgpt2"
PROMPT = "Caption:"
DEFAULT_SAVE_PATH = "common/outputs/day5_real_vlm/prefix_projector.pt"


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad_(False)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


class ClipPrefixCaptioner(nn.Module):
    def __init__(
        self,
        clip_model: nn.Module,
        language_model: nn.Module,
        prefix_length: int = 8,
        projector_hidden: int = 1024,
    ):
        super().__init__()
        self.clip_model = clip_model
        self.language_model = language_model
        self.prefix_length = prefix_length

        vision_dim = clip_model.config.projection_dim
        llm_dim = getattr(language_model.config, "n_embd", None)
        if llm_dim is None:
            llm_dim = language_model.config.hidden_size

        self.prefix_projector = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, projector_hidden),
            nn.GELU(),
            nn.Linear(projector_hidden, prefix_length * llm_dim),
        )
        self.prefix_norm = nn.LayerNorm(llm_dim)

        freeze(self.clip_model)
        freeze(self.language_model)
        self.clip_model.eval()
        self.language_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.clip_model.eval()
        self.language_model.eval()
        return self

    def encode_image_prefix(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(pixel_values=pixel_values)
            image_features = F.normalize(image_features, dim=-1)

        prefix = self.prefix_projector(image_features)
        prefix = prefix.view(pixel_values.shape[0], self.prefix_length, -1)
        return self.prefix_norm(prefix)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        trace: bool = False,
    ):
        prefix_embeds = self.encode_image_prefix(pixel_values)
        token_embeds = self.language_model.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        if trace:
            print("pixel_values:", tuple(pixel_values.shape))
            print("visual_prefix:", tuple(prefix_embeds.shape))
            print("text_input_ids:", tuple(input_ids.shape))
            print("text_embeds:", tuple(token_embeds.shape))
            print("llm_inputs_embeds:", tuple(inputs_embeds.shape))
            print("labels:", tuple(labels.shape))

        return self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    @torch.no_grad()
    def generate_caption(
        self,
        pixel_values: torch.Tensor,
        tokenizer: Any,
        device: str,
        max_new_tokens: int = 30,
    ) -> str:
        self.eval()
        pixel_values = pixel_values.to(device)
        prefix_embeds = self.encode_image_prefix(pixel_values)

        prompt_ids = tokenizer(PROMPT, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            if generated:
                generated_ids = torch.tensor([generated], dtype=torch.long, device=device)
                text_ids = torch.cat([prompt_ids, generated_ids], dim=1)
            else:
                text_ids = prompt_ids

            token_embeds = self.language_model.get_input_embeddings()(text_ids)
            inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)
            attention_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device)

            logits = self.language_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
            next_id = logits[:, -1].argmax(dim=-1).item()

            if next_id == tokenizer.eos_token_id:
                break
            generated.append(next_id)

        return tokenizer.decode(generated, skip_special_tokens=True).strip()


class FlickrCaptionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], caption_mode: str = "random"):
        self.rows = rows
        self.caption_mode = caption_mode

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        captions = row["caption"]
        if isinstance(captions, str):
            caption = captions
        elif self.caption_mode == "random":
            caption = random.choice(captions)
        else:
            caption = captions[0]

        return {
            "image": row["image"].convert("RGB"),
            "caption": caption.strip(),
            "filename": row.get("filename", str(idx)),
        }


@dataclass
class CaptionCollator:
    clip_processor: Any
    tokenizer: Any
    prefix_length: int
    max_caption_tokens: int

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        images = [example["image"] for example in examples]
        captions = [example["caption"] for example in examples]

        pixel_values = self.clip_processor(images=images, return_tensors="pt").pixel_values

        prompt_ids = self.tokenizer(PROMPT, add_special_tokens=False).input_ids
        text_ids_list: list[list[int]] = []
        text_mask_list: list[list[int]] = []
        label_list: list[list[int]] = []

        for caption in captions:
            caption_ids = self.tokenizer(" " + caption, add_special_tokens=False).input_ids
            caption_ids = caption_ids[: max(self.max_caption_tokens - 1, 1)]
            caption_ids.append(self.tokenizer.eos_token_id)

            text_ids = prompt_ids + caption_ids
            labels = [-100] * (self.prefix_length + len(prompt_ids)) + caption_ids

            text_ids_list.append(text_ids)
            text_mask_list.append([1] * len(text_ids))
            label_list.append(labels)

        max_text_len = max(len(ids) for ids in text_ids_list)
        max_label_len = self.prefix_length + max_text_len

        for ids, mask, labels in zip(text_ids_list, text_mask_list, label_list):
            text_pad = max_text_len - len(ids)
            label_pad = max_label_len - len(labels)
            ids.extend([self.tokenizer.pad_token_id] * text_pad)
            mask.extend([0] * text_pad)
            labels.extend([-100] * label_pad)

        text_attention = torch.tensor(text_mask_list, dtype=torch.long)
        prefix_attention = torch.ones(len(examples), self.prefix_length, dtype=torch.long)

        return {
            "pixel_values": pixel_values,
            "input_ids": torch.tensor(text_ids_list, dtype=torch.long),
            "attention_mask": torch.cat([prefix_attention, text_attention], dim=1),
            "labels": torch.tensor(label_list, dtype=torch.long),
            "captions": captions,
            "filenames": [example["filename"] for example in examples],
        }


def require_hf_libraries():
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor
    except ImportError as exc:
        raise SystemExit(
            "缺少依赖。请先运行: .venv/bin/python -m pip install -r requirements.txt"
        ) from exc

    return load_dataset, AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor


def load_flickr30k_rows(
    dataset_name: str,
    target_split: str,
    max_samples: int,
    seed: int,
    streaming: bool,
) -> list[dict[str, Any]]:
    load_dataset, *_ = require_hf_libraries()
    dataset = load_dataset(dataset_name, split="test", streaming=streaming)

    if streaming:
        rows = []
        for row in dataset:
            if row.get("split") != target_split:
                continue
            rows.append(row)
            if len(rows) >= max_samples:
                break
        return rows

    dataset = dataset.filter(lambda row: row["split"] == target_split)
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(max_samples, len(dataset))))
    return [dataset[i] for i in range(len(dataset))]


def build_model_and_processors(args: argparse.Namespace, device: str):
    _, AutoModelForCausalLM, AutoTokenizer, CLIPModel, CLIPProcessor = require_hf_libraries()

    clip_processor = CLIPProcessor.from_pretrained(args.clip_model)
    clip_model = CLIPModel.from_pretrained(args.clip_model)

    tokenizer = AutoTokenizer.from_pretrained(args.llm_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    language_model = AutoModelForCausalLM.from_pretrained(args.llm_model)
    model = ClipPrefixCaptioner(
        clip_model=clip_model,
        language_model=language_model,
        prefix_length=args.prefix_length,
        projector_hidden=args.projector_hidden,
    ).to(device)

    return model, clip_processor, tokenizer


def trainable_state_dict(model: ClipPrefixCaptioner, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prefix_projector": model.prefix_projector.state_dict(),
        "prefix_norm": model.prefix_norm.state_dict(),
        "prefix_length": model.prefix_length,
        "clip_model": args.clip_model,
        "llm_model": args.llm_model,
    }


def save_trainable_checkpoint(model: ClipPrefixCaptioner, path: str, args: argparse.Namespace) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trainable_state_dict(model, args), output_path)
    print(f"saved projector checkpoint: {output_path}")


def load_trainable_checkpoint(model: ClipPrefixCaptioner, path: str) -> None:
    if not path:
        return
    checkpoint = torch.load(path, map_location="cpu")
    model.prefix_projector.load_state_dict(checkpoint["prefix_projector"])
    model.prefix_norm.load_state_dict(checkpoint["prefix_norm"])
    print(f"loaded projector checkpoint: {path}")


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def token_nll_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels.ne(-100)

    safe_labels = shifted_labels.masked_fill(~valid, 0)
    losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        safe_labels.view(-1),
        reduction="none",
    ).view_as(safe_labels)

    losses = losses * valid
    return losses.sum(dim=1) / valid.sum(dim=1).clamp_min(1)


@torch.no_grad()
def evaluate_loss(model: ClipPrefixCaptioner, loader: DataLoader, device: str, max_batches: int) -> dict[str, float]:
    was_training = model.training
    model.eval()

    total_loss = 0.0
    total_count = 0

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        batch_size = batch["input_ids"].shape[0]
        total_loss += outputs.loss.item() * batch_size
        total_count += batch_size

        if step >= max_batches:
            break

    if was_training:
        model.train()

    mean_loss = total_loss / max(total_count, 1)
    return {"loss": mean_loss, "ppl": math.exp(min(mean_loss, 20.0))}


@torch.no_grad()
def caption_rank_probe(
    model: ClipPrefixCaptioner,
    rows: list[dict[str, Any]],
    collator: CaptionCollator,
    device: str,
    max_items: int,
) -> tuple[int, int]:
    was_training = model.training
    model.eval()

    rows = rows[:max_items]
    captions = [row["caption"][0] if isinstance(row["caption"], list) else row["caption"] for row in rows]
    correct = 0

    for image_index, image_row in enumerate(rows):
        candidates = []
        for caption in captions:
            candidates.append(
                {
                    "image": image_row["image"].convert("RGB"),
                    "caption": caption,
                    "filename": image_row.get("filename", str(image_index)),
                }
            )

        batch = move_batch(collator(candidates), device)
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        nll = token_nll_from_logits(outputs.logits, batch["labels"])
        correct += int(nll.argmin().item() == image_index)

    if was_training:
        model.train()

    return correct, len(rows)


def train_one_epoch(
    model: ClipPrefixCaptioner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    log_every: int,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0

    for step, batch in enumerate(loader, start=1):
        batch = move_batch(batch, device)
        outputs = model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        optimizer.zero_grad(set_to_none=True)
        outputs.loss.backward()
        optimizer.step()

        batch_size = batch["input_ids"].shape[0]
        total_loss += outputs.loss.item() * batch_size
        total_count += batch_size

        if step % log_every == 0:
            print(f"  step {step:04d} | loss={outputs.loss.item():.4f}")

    return total_loss / max(total_count, 1)


@torch.no_grad()
def show_generations(
    title: str,
    model: ClipPrefixCaptioner,
    rows: list[dict[str, Any]],
    collator: CaptionCollator,
    tokenizer: Any,
    device: str,
    max_new_tokens: int,
    count: int = 3,
) -> None:
    print(title)
    model.eval()
    examples = FlickrCaptionDataset(rows[:count], caption_mode="first")
    for idx in range(min(count, len(examples))):
        example = examples[idx]
        batch = move_batch(collator([example]), device)
        generated = model.generate_caption(
            pixel_values=batch["pixel_values"],
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        print(f"  image={example['filename']}")
        print(f"    target: {example['caption']}")
        print(f"    pred:   {generated}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--clip-model", default=DEFAULT_CLIP)
    parser.add_argument("--llm-model", default=DEFAULT_LLM)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--projector-hidden", type=int, default=1024)
    parser.add_argument("--max-caption-tokens", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--rank-probe-size", type=int, default=6)
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

    device = args.device or get_default_device()

    print("目标: Day5 real VLM = frozen CLIP + trainable prefix projector + frozen DistilGPT2")
    print(f"dataset: {args.dataset} (Flickr30k image-caption pairs)")
    print(f"clip: {args.clip_model}")
    print(f"llm: {args.llm_model}")
    print(f"device: {device}")
    print()

    train_rows = load_flickr30k_rows(args.dataset, "train", args.train_samples, args.seed, args.streaming)
    val_rows = load_flickr30k_rows(args.dataset, "test", args.val_samples, args.seed + 1, args.streaming)

    model, clip_processor, tokenizer = build_model_and_processors(args, device)
    load_trainable_checkpoint(model, args.load_path)
    collator = CaptionCollator(
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        prefix_length=args.prefix_length,
        max_caption_tokens=args.max_caption_tokens,
    )

    train_loader = DataLoader(
        FlickrCaptionDataset(train_rows, caption_mode="random"),
        batch_size=args.batch_size,
        shuffle=not args.streaming,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        FlickrCaptionDataset(val_rows, caption_mode="first"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
    )

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    total, trainable = count_parameters(model)

    print(f"parameters: total={total:,}, trainable={trainable:,}")
    print("trainable modules: prefix_projector, prefix_norm")
    print()

    first = move_batch(next(iter(train_loader)), device)
    with torch.inference_mode():
        model(
            pixel_values=first["pixel_values"],
            input_ids=first["input_ids"],
            attention_mask=first["attention_mask"],
            labels=first["labels"],
            trace=True,
        )
    print()

    show_generations("训练前 generation:", model, val_rows, collator, tokenizer, device, args.max_new_tokens)
    before_eval = evaluate_loss(model, val_loader, device, args.eval_batches)
    before_rank = caption_rank_probe(model, val_rows, collator, device, args.rank_probe_size)
    print(f"训练前 eval: loss={before_eval['loss']:.4f} ppl={before_eval['ppl']:.2f}")
    print(f"训练前 caption-rank probe: {before_rank[0]}/{before_rank[1]}")
    print()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, args.log_every)
        val_eval = evaluate_loss(model, val_loader, device, args.eval_batches)
        rank = caption_rank_probe(model, val_rows, collator, device, args.rank_probe_size)
        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_eval['loss']:.4f} | val_ppl={val_eval['ppl']:.2f} | "
            f"rank={rank[0]}/{rank[1]}"
        )

    print()
    show_generations("训练后 generation:", model, val_rows, collator, tokenizer, device, args.max_new_tokens)
    save_trainable_checkpoint(model, args.save_path, args)


if __name__ == "__main__":
    main()
