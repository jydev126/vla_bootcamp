from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.driving_toy import ensure_data, future_to_tensor


def continuous_to_action_ids(target: torch.Tensor, bins: int) -> torch.Tensor:
    return torch.round((target.clamp(-1, 1) + 1.0) * 0.5 * (bins - 1)).long().clamp(0, bins - 1)


def action_ids_to_continuous(ids: torch.Tensor, bins: int) -> torch.Tensor:
    return ids.float() / max(bins - 1, 1) * 2.0 - 1.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=int, default=32)
    args = parser.parse_args()
    sample = ensure_data(num_samples=4)[0]
    target = future_to_tensor(sample)
    ids = continuous_to_action_ids(target, args.bins)
    restored = action_ids_to_continuous(ids, args.bins)
    print("source = OpenVLA action tokenization idea")
    print("continuous trajectory:", target.shape, target[:6])
    print("action token ids:", ids.shape, ids[:6])
    print("detokenized approx:", restored.shape, restored[:6])


if __name__ == "__main__":
    main()
