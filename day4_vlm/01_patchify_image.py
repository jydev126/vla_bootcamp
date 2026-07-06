from __future__ import annotations

"""
Day 4 / 01

把一张图片切成很多小块 patch。

为什么这一步重要？

GPT 的输入是 token:
    [B, T]

ViT / VLM 也想把图片变成 token:
    image -> patches -> patch embeddings -> visual tokens

这份代码只做第一步:
    [B, 3, 64, 64] -> [B, 64, 192]

如果 patch_size = 8:
    一张 64x64 图片会被切成 8x8 = 64 个 patch
    每个 patch 是 3x8x8 = 192 个数字
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.driving_toy import IMAGE_SIZE, ensure_data, image_to_tensor, render_bev


def patchify(pixel_values: torch.Tensor, patch_size: int) -> torch.Tensor:
    """
    pixel_values:
        [B, C, H, W]

    return:
        [B, num_patches, C * patch_size * patch_size]

    核心直觉:
        每个 patch 就像 GPT 里的一个 token。
        只是 GPT token 是一个整数，image patch 是一小块像素。
    """
    B, C, H, W = pixel_values.shape

    if H % patch_size != 0 or W % patch_size != 0:
        raise ValueError("H and W must be divisible by patch_size")

    # unfold 会沿着高、宽两个方向滑窗。
    # 因为 step = patch_size，所以 patch 之间不重叠。
    patches = pixel_values.unfold(2, patch_size, patch_size)
    patches = patches.unfold(3, patch_size, patch_size)

    # 当前形状:
    # [B, C, H/patch, W/patch, patch_size, patch_size]
    #
    # 我们希望 patch 维度排在前面:
    # [B, H/patch, W/patch, C, patch_size, patch_size]
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()

    # 最后把每个小块摊平成一个向量。
    return patches.view(B, -1, C * patch_size * patch_size)


def unpatchify(patches: torch.Tensor, patch_size: int, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """
    patchify 的反操作。

    这个函数不是 VLM 必需的，但它能帮你确认:
    patchify 只是重新排列像素，没有丢信息。
    """
    B, num_patches, flat_dim = patches.shape
    C = flat_dim // (patch_size * patch_size)
    grid = image_size // patch_size

    if num_patches != grid * grid:
        raise ValueError("num_patches does not match image_size and patch_size")

    x = patches.view(B, grid, grid, C, patch_size, patch_size)
    x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
    return x.view(B, C, image_size, image_size)


def print_one_patch(patches: torch.Tensor, patch_id: int) -> None:
    """
    打印一个 patch 的统计量。

    不直接打印 192 个数字，因为那只会制造噪音。
    """
    patch = patches[0, patch_id]
    print(f"patch[{patch_id}] shape:", patch.shape)
    print(f"patch[{patch_id}] min/max:", round(patch.min().item(), 3), round(patch.max().item(), 3))
    print(f"patch[{patch_id}] mean:", round(patch.mean().item(), 3))
    print(f"patch[{patch_id}] first 12 values:", [round(x, 3) for x in patch[:12].tolist()])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-size", type=int, default=8)
    args = parser.parse_args()

    sample = ensure_data(num_samples=4)[0]
    image = render_bev(sample)
    pixel_values = image_to_tensor(image).unsqueeze(0)

    patches = patchify(pixel_values, args.patch_size)
    rebuilt = unpatchify(patches, args.patch_size)
    max_error = (pixel_values - rebuilt).abs().max().item()

    grid = IMAGE_SIZE // args.patch_size

    print("目标: image -> patches")
    print("sample command/reason:", sample["command"], sample["reason"])
    print("pixel_values:", tuple(pixel_values.shape), "= [B, C, H, W]")
    print("patch grid:", f"{grid} x {grid}")
    print("patches:", tuple(patches.shape), "= [B, num_patches, patch_dim]")
    print("one patch dim:", patches.shape[-1], "= C * patch_size * patch_size")
    print("rebuild max error:", max_error)
    print()

    print_one_patch(patches, patch_id=0)
    print()
    print_one_patch(patches, patch_id=patches.shape[1] // 2)


if __name__ == "__main__":
    main()
