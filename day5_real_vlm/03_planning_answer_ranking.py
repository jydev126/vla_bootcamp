from __future__ import annotations

"""
Day 5 / 03

驾驶 VLM 不能只会自由生成一句 caption。
很多自动驾驶接口更像是在候选动作/风险解释之间做选择:

    image + question + candidate answer -> answer loss

如果正确候选的 loss 更低，说明视觉 prefix 正在影响语言模型对答案的偏好。
这个脚本加载 02 训练出的 projector，做 planning-style answer ranking probe。
"""

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

SCRIPT2_PATH = Path(__file__).with_name("02_driving_qa_tuning.py")
DEFAULT_LOAD_PATH = "common/outputs/day5_real_vlm/driving_qa_projector.pt"


def load_script2():
    spec = importlib.util.spec_from_file_location("day5_driving_qa_tuning", SCRIPT2_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


script2 = load_script2()
core = script2.core


def candidate_set() -> list[str]:
    return [
        "continue carefully",
        "keep a safe distance from vehicles",
        "slow down and watch for pedestrians",
        "slow down and give the cyclist space",
    ]


@torch.no_grad()
def score_candidates(
    model: core.ClipPrefixCaptioner,
    row: dict[str, Any],
    collator: script2.DrivingQACollator,
    device: str,
) -> list[tuple[str, float]]:
    question = "What is a reasonable high level driving response?"
    examples = []
    for answer in candidate_set():
        examples.append(
            {
                "image": row["image"].convert("RGB"),
                "question": question,
                "answer": answer,
                "kind": "planning",
                "filename": row.get("filename", "unknown"),
            }
        )

    batch = core.move_batch(collator(examples), device)
    outputs = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )
    losses = core.token_nll_from_logits(outputs.logits, batch["labels"]).detach().cpu().tolist()
    return sorted(zip(candidate_set(), losses), key=lambda item: item[1])


def explain_tags(tags: dict[str, bool]) -> str:
    active = [name for name, value in tags.items() if value]
    return ", ".join(active) if active else "none"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=core.DEFAULT_DATASET)
    parser.add_argument("--clip-model", default=core.DEFAULT_CLIP)
    parser.add_argument("--llm-model", default=core.DEFAULT_LLM)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--prefix-length", type=int, default=8)
    parser.add_argument("--projector-hidden", type=int, default=1024)
    parser.add_argument("--max-answer-tokens", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--load-path", default=DEFAULT_LOAD_PATH)
    args = parser.parse_args()

    device = args.device or core.get_default_device()

    print("目标: planning-style answer ranking probe")
    print("同一张图下，不自由生成长文本，而是比较候选驾驶响应的 token loss")
    print(f"dataset: {args.dataset} filtered by driving terms")
    print(f"load projector: {args.load_path or '(random bridge)'}")
    print(f"device: {device}")
    print()

    rows = script2.load_driving_rows(args.dataset, "test", args.samples, args.seed, args.streaming)
    if not rows:
        raise SystemExit("没有筛到 driving rows。可以增大 --samples 或去掉 --streaming 后重试。")

    model, clip_processor, tokenizer = core.build_model_and_processors(args, device)
    if args.load_path and Path(args.load_path).exists():
        core.load_trainable_checkpoint(model, args.load_path)
    elif args.load_path:
        print(f"checkpoint not found, using random bridge: {args.load_path}")
    collator = script2.DrivingQACollator(
        clip_processor=clip_processor,
        tokenizer=tokenizer,
        prefix_length=args.prefix_length,
        max_answer_tokens=args.max_answer_tokens,
    )
    model.eval()

    correct = 0
    total = 0

    for idx, row in enumerate(rows):
        tags = script2.scene_tags(row)
        target = script2.action_answer(tags)
        ranked = score_candidates(model, row, collator, device)
        pred = ranked[0][0]
        correct += int(pred == target)
        total += 1

        print(f"sample {idx:02d} image={row.get('filename', idx)}")
        print(f"  tags:   {explain_tags(tags)}")
        print(f"  target: {target}")
        for answer, loss in ranked:
            marker = "*" if answer == target else " "
            print(f"  {marker} loss={loss:.3f} answer={answer}")
        print()

    print(f"planning-rank accuracy: {correct}/{total}")
    print()
    print("读这个 probe 的方式:")
    print("  如果正确候选 loss 更低，说明视觉 prefix 帮 LLM 偏向了合适的驾驶回答。")
    print("  如果所有样本都偏向同一个答案，说明模型更多在靠语言先验，而不是看图。")


if __name__ == "__main__":
    main()
