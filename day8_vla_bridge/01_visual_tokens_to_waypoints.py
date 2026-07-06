from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.driving_toy import build_vocab, encode_tokens, ensure_data, future_to_tensor, image_to_tensor, render_bev, split_samples, state_tokens


def load_bridge():
    path = ROOT / "day4_vlm" / "03_minigpt4_bridge.py"
    spec = importlib.util.spec_from_file_location("minigpt4_bridge", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


bridge = load_bridge()
MAX_TEXT_LEN = 16


class PixelActionDataset(Dataset):
    def __init__(self, samples, vocab):
        self.samples = samples
        self.vocab = vocab

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ids, mask = encode_tokens(state_tokens(sample), self.vocab, MAX_TEXT_LEN)
        return image_to_tensor(render_bev(sample)), ids, mask, future_to_tensor(sample)


class VisualWaypointVLA(nn.Module):
    def __init__(self, vocab_size, num_latents=4):
        super().__init__()
        self.vision_encoder = bridge.TinyViTEncoder()
        self.projector = bridge.VisionProjector()
        self.text_embedding = nn.Embedding(vocab_size, 96, padding_idx=0)
        self.latents = nn.Parameter(torch.randn(1, num_latents, 96) * 0.02)
        self.pos = nn.Embedding(64 + MAX_TEXT_LEN + num_latents, 96)
        layer = nn.TransformerEncoderLayer(96, 4, 384, batch_first=True, activation="gelu")
        self.fusion = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(96), nn.Linear(96, 128), nn.GELU(), nn.Linear(128, 12))

    def forward(self, image, ids, mask, trace=False):
        vf, patches = self.vision_encoder(image)
        visual = self.projector(vf)
        text = self.text_embedding(ids)
        lat = self.latents.expand(image.shape[0], -1, -1)
        tokens = torch.cat([visual, text, lat], dim=1)
        pos = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.fusion(tokens + self.pos(pos))
        latent_h = hidden[:, -lat.shape[1]:]
        traj = self.head(latent_h.mean(dim=1))
        if trace:
            print("image_patches:", patches.shape)
            print("visual_tokens:", visual.shape)
            print("text_tokens:", text.shape)
            print("latent_tokens:", lat.shape)
            print("latent_h:", latent_h.shape)
            print("trajectory:", traj.shape)
        return traj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    samples = ensure_data(num_samples=args.num_samples)
    train, _ = split_samples(samples)
    vocab = build_vocab(train)
    loader = DataLoader(PixelActionDataset(train, vocab), batch_size=args.batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisualWaypointVLA(len(vocab)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    first = next(iter(loader))
    print("Day8: VLM hidden state becomes action head input")
    model(first[0].to(device), first[1].to(device), first[2].to(device), trace=True)
    for epoch in range(1, args.epochs + 1):
        total = 0.0
        count = 0
        for image, ids, mask, y in loader:
            loss = loss_fn(model(image.to(device), ids.to(device), mask.to(device)), y.to(device))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * image.shape[0]; count += image.shape[0]
        print(f"epoch {epoch:02d} loss={total / count:.6f}")


if __name__ == "__main__":
    main()
