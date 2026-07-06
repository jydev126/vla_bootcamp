from __future__ import annotations

"""
Day 4 / 02

把 patch 变成 visual token。

上一节:
    image -> patches

这一节:
    patches -> patch embeddings -> Transformer encoder -> visual features

这就是一个极小版 ViT encoder。
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.driving_toy import IMAGE_SIZE, ensure_data, image_to_tensor, render_bev


def load_patchify():
    path = Path(__file__).with_name("01_patchify_image.py")
    spec = importlib.util.spec_from_file_location("patchify_image", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module.patchify


patchify = load_patchify()


class TinyViTEncoder(nn.Module):
    """
    一个最小 ViT encoder。

    输入:
        pixel_values: [B, 3, 64, 64]

    输出:
        visual_features: [B, 64, vision_dim]

    64 表示 64 个图片 patch。
    vision_dim 表示每个 patch token 的隐藏维度。
    """

    def __init__(self, patch_size: int = 8, vision_dim: int = 64, image_size: int = IMAGE_SIZE):
        super().__init__()

        self.patch_size = patch_size
        self.vision_dim = vision_dim
        self.num_patches = (image_size // patch_size) ** 2

        patch_dim = 3 * patch_size * patch_size

        # 像 GPT 的 token embedding。
        # 区别是: GPT 查表，ViT 用线性层把像素向量投影成 embedding。
        self.patch_projection = nn.Linear(patch_dim, vision_dim)

        # 像 GPT 的 position embedding。
        # 图片 patch 也需要位置信息，否则模型不知道哪个 patch 在左上角。
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, vision_dim) * 0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=vision_dim,
            nhead=4,
            dim_feedforward=vision_dim * 4,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        patches = patchify(pixel_values, self.patch_size)

        patch_tokens = self.patch_projection(patches)
        patch_tokens = patch_tokens + self.position_embedding

        visual_features = self.encoder(patch_tokens)

        return visual_features, patch_tokens, patches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    torch.manual_seed(42)

    samples = ensure_data(num_samples=args.batch_size)[: args.batch_size]
    pixel_values = torch.stack([image_to_tensor(render_bev(sample)) for sample in samples])

    model = TinyViTEncoder()
    visual_features, patch_tokens, patches = model(pixel_values)

    print("目标: patches -> visual tokens")
    print("pixel_values:", tuple(pixel_values.shape))
    print("patches:", tuple(patches.shape))
    print("patch_tokens after Linear + position:", tuple(patch_tokens.shape))
    print("visual_features after Transformer:", tuple(visual_features.shape))
    print()
    print("第 0 张图的第 0 个 visual token 前 8 维:")
    print([round(x, 3) for x in visual_features[0, 0, :8].tolist()])
    print()
    print("直觉:")
    print("patch_tokens 是每个小图块自己的表示。")
    print("visual_features 是经过 self-attention 后的表示，每个 patch 已经看过其它 patch。")


if __name__ == "__main__":
    main()
