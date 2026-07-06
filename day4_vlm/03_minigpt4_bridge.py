from __future__ import annotations

"""
Day 4 / 03

把 visual token 接到 tiny language model。

MiniGPT-4 的核心桥接思想可以简化成一句话:

    frozen vision encoder -> projector -> LLM hidden space

vision encoder 输出的维度叫 vision_dim。
LLM 使用的 hidden size 叫 llm_dim。

projector 的工作就是:
    [B, num_visual_tokens, vision_dim]
        -> [B, num_visual_tokens, llm_dim]

这样 visual token 才能和 text token 拼在同一个序列里。
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.driving_toy import ensure_data, image_to_tensor, render_bev


def load_tiny_vit():
    path = Path(__file__).with_name("02_tiny_vit_encoder.py")
    spec = importlib.util.spec_from_file_location("tiny_vit_encoder", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.TinyViTEncoder


TinyViTEncoder = load_tiny_vit()


class VisionProjector(nn.Module):
    """
    把 vision_dim 对齐到 llm_dim。

    真实 MiniGPT-4 用 Q-Former / projection 等桥接模块。
    这里先用最容易看懂的 LayerNorm + Linear。
    """

    def __init__(self, vision_dim: int = 64, llm_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(vision_dim),
            nn.Linear(vision_dim, llm_dim),
        )

    def forward(self, visual_features: torch.Tensor) -> torch.Tensor:
        return self.net(visual_features)


class TinyTextBackbone(nn.Module):
    """
    一个极小 text Transformer。

    它不负责把图片变成 token。
    它只接收已经在 llm_dim 空间里的 embedding。
    """

    def __init__(self, vocab_size: int, llm_dim: int = 96, max_len: int = 96):
        super().__init__()

        self.llm_dim = llm_dim
        self.token_embedding = nn.Embedding(vocab_size, llm_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, llm_dim)

        layer = nn.TransformerEncoderLayer(
            d_model=llm_dim,
            nhead=4,
            dim_feedforward=llm_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=2)
        self.final_norm = nn.LayerNorm(llm_dim)

    def forward(self, embedded_tokens: torch.Tensor) -> torch.Tensor:
        B, T, _ = embedded_tokens.shape

        pos = torch.arange(T, device=embedded_tokens.device)
        pos = pos.unsqueeze(0).expand(B, T)

        x = embedded_tokens + self.position_embedding(pos)
        x = self.layers(x)
        return self.final_norm(x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(42)

    vocab = {"PAD": 0, "BOS": 1, "what": 2, "affects": 3, "ego": 4, "planning": 5, "ANSWER": 6}
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]] * args.batch_size)

    samples = ensure_data(num_samples=args.batch_size)[: args.batch_size]
    pixel_values = torch.stack([image_to_tensor(render_bev(sample)) for sample in samples])

    vision_encoder = TinyViTEncoder()
    projector = VisionProjector()
    tiny_llm = TinyTextBackbone(vocab_size=len(vocab))

    visual_features, patch_tokens, patches = vision_encoder(pixel_values)
    visual_tokens = projector(visual_features)
    text_tokens = tiny_llm.token_embedding(input_ids)

    # 这一步就是 VLM 的关键拼接:
    # 图片 token 放前面，文字 token 放后面。
    vlm_inputs = torch.cat([visual_tokens, text_tokens], dim=1)
    hidden = tiny_llm(vlm_inputs)

    answer_position = visual_tokens.shape[1] + input_ids.shape[1] - 1
    answer_hidden = hidden[:, answer_position]

    print("目标: visual tokens + text tokens -> one LLM sequence")
    print("patches:", tuple(patches.shape))
    print("patch_tokens:", tuple(patch_tokens.shape))
    print("visual_features:", tuple(visual_features.shape), "vision_dim = 64")
    print("visual_tokens after projector:", tuple(visual_tokens.shape), "llm_dim = 96")
    print("text_tokens:", tuple(text_tokens.shape), "llm_dim = 96")
    print("vlm_inputs:", tuple(vlm_inputs.shape), "= [visual tokens, text tokens]")
    print("llm_hidden:", tuple(hidden.shape))
    print("answer_hidden:", tuple(answer_hidden.shape))
    print()
    print("直觉:")
    print("projector 不是负责回答问题的模块。")
    print("它只负责把图像特征翻译成 LLM 能接住的 token embedding。")


if __name__ == "__main__":
    main()
