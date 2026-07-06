from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageDraw


COMMANDS = ["keep", "brake", "left", "right"]
REASONS = ["keep", "brake_due_to_close_lead", "lane_change_left", "lane_change_right"]
DATA_PATH = Path("common/outputs/toy_driving.jsonl")
IMAGE_SIZE = 64


def reason_from_sample(sample: dict) -> str:
    if sample["command"] == "brake" or (sample["lead_distance"] < 14.0 and sample["rel_speed"] < -1.0):
        return "brake_due_to_close_lead"
    if sample["command"] == "left":
        return "lane_change_left"
    if sample["command"] == "right":
        return "lane_change_right"
    return "keep"


def generate_sample(rng: random.Random) -> dict:
    command = rng.choice(COMMANDS)
    ego_speed = rng.uniform(3.0, 14.0)
    lead_distance = rng.uniform(6.0, 40.0)
    rel_speed = rng.uniform(-5.0, 3.0)
    history = []
    for t in [-1.5, -1.0, -0.5]:
        history.append([round(ego_speed * t + rng.gauss(0.0, 0.08), 3), round(rng.gauss(0.0, 0.04), 3)])

    speed_scale = ego_speed / 10.0
    close_factor = 1.0
    if lead_distance < 16.0 and rel_speed < 0.0:
        close_factor = max(0.35, lead_distance / 16.0 + rel_speed / 12.0)

    future = []
    for step in range(1, 7):
        if command == "brake":
            dx = speed_scale * sum(max(0.25, 1.0 - 0.12 * k) for k in range(1, step + 1))
            dx *= min(close_factor, 0.75)
            y = 0.0
        elif command == "left":
            dx = speed_scale * step * close_factor
            y = 0.18 * step + 0.035 * step * step
        elif command == "right":
            dx = speed_scale * step * close_factor
            y = -0.18 * step - 0.035 * step * step
        else:
            dx = speed_scale * step * close_factor
            y = 0.0
        future.append([round(dx + rng.gauss(0.0, 0.04), 3), round(y + rng.gauss(0.0, 0.025), 3)])

    sample = {
        "ego_speed": round(ego_speed, 3),
        "lead_distance": round(lead_distance, 3),
        "rel_speed": round(rel_speed, 3),
        "command": command,
        "history": history,
        "future": future,
    }
    sample["reason"] = reason_from_sample(sample)
    return sample


def write_jsonl(samples: Iterable[dict], path: Path = DATA_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")


def load_jsonl(path: Path = DATA_PATH) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ensure_data(path: Path = DATA_PATH, num_samples: int = 10000, seed: int = 42) -> list[dict]:
    if path.exists():
        samples = load_jsonl(path)
        if len(samples) >= num_samples:
            return samples[:num_samples]
    rng = random.Random(seed)
    samples = [generate_sample(rng) for _ in range(num_samples)]
    write_jsonl(samples, path)
    return samples


def split_samples(samples: list[dict], val_ratio: float = 0.15) -> tuple[list[dict], list[dict]]:
    n_val = max(1, int(len(samples) * val_ratio))
    return samples[:-n_val], samples[-n_val:]


def future_to_tensor(sample: dict) -> torch.Tensor:
    flat = []
    for x, y in sample["future"]:
        flat.extend([x / 10.0, y / 4.0])
    return torch.tensor(flat, dtype=torch.float32)


def future_world_target(sample: dict) -> torch.Tensor:
    future_distance = max(0.0, min(45.0, sample["lead_distance"] + sample["rel_speed"]))
    return torch.tensor([future_distance / 45.0], dtype=torch.float32)


def make_numeric_token(value: float, low: float, high: float, bins: int, prefix: str) -> str:
    ratio = (value - low) / max(high - low, 1e-6)
    idx = int(max(0, min(bins - 1, math.floor(ratio * bins))))
    return f"{prefix}_{idx}"


def state_tokens(sample: dict) -> list[str]:
    tokens = [
        "BOS",
        make_numeric_token(sample["ego_speed"], 0.0, 16.0, 8, "speed"),
        make_numeric_token(sample["lead_distance"], 0.0, 45.0, 9, "dist"),
        make_numeric_token(sample["rel_speed"], -6.0, 4.0, 8, "rel"),
        f"cmd_{sample['command']}",
    ]
    for i, (x, y) in enumerate(sample["history"]):
        tokens.append(make_numeric_token(x, -25.0, 2.0, 8, f"hist{i}_x"))
        tokens.append(make_numeric_token(y, -4.0, 4.0, 5, f"hist{i}_y"))
    tokens.append("EOS")
    return tokens


def build_vocab(samples: list[dict], extra_tokens: Iterable[str] = ()) -> dict[str, int]:
    vocab = {"PAD": 0, "UNK": 1}
    for token in extra_tokens:
        if token not in vocab:
            vocab[token] = len(vocab)
    for sample in samples:
        for token in state_tokens(sample):
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab


def encode_tokens(tokens: list[str], vocab: dict[str, int], max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    ids = [vocab.get(token, vocab["UNK"]) for token in tokens[:max_len]]
    mask = [1] * len(ids)
    while len(ids) < max_len:
        ids.append(vocab["PAD"])
        mask.append(0)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(mask, dtype=torch.bool)


def render_bev(sample: dict, image_size: int = IMAGE_SIZE) -> Image.Image:
    image = Image.new("RGB", (image_size, image_size), color=(44, 48, 52))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 0, 52, image_size), fill=(76, 78, 80))
    draw.line((32, 0, 32, image_size), fill=(232, 218, 80), width=2)
    draw.rectangle((27, 50, 37, 60), fill=(245, 245, 245))
    lead_y = int(50 - sample["lead_distance"] / 45.0 * 44.0)
    lead_y = max(4, min(46, lead_y))
    color = (210, 54, 48) if sample["rel_speed"] < -1.0 else (55, 100, 210)
    draw.rectangle((27, lead_y, 37, lead_y + 8), fill=color)
    if sample["command"] == "left":
        draw.polygon([(18, 14), (10, 20), (18, 26)], fill=(90, 220, 130))
    elif sample["command"] == "right":
        draw.polygon([(46, 14), (54, 20), (46, 26)], fill=(90, 220, 130))
    elif sample["command"] == "brake":
        draw.rectangle((24, 8, 40, 14), fill=(245, 180, 70))
    return image


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    values = torch.tensor(bytearray(image.tobytes()), dtype=torch.float32)
    values = values.view(image.height, image.width, 3) / 255.0
    return values.permute(2, 0, 1)
