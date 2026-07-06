from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.driving_toy import REASONS, build_vocab, encode_tokens, ensure_data, future_to_tensor, future_world_target, reason_from_sample, split_samples, state_tokens

MAX_LEN = 32


def latent_tokens(sample, lang_latents, vis_latents):
    tokens = state_tokens(sample)[:-1]
    tokens += [f"LANG_LATENT_{i}" for i in range(lang_latents)]
    tokens += [f"VIS_LATENT_{i}" for i in range(vis_latents)]
    tokens.append("EOS")
    return tokens


class OneVLToyDataset(Dataset):
    def __init__(self, samples, vocab, lang_latents, vis_latents):
        self.samples = samples
        self.vocab = vocab
        self.lang_latents = lang_latents
        self.vis_latents = vis_latents
        self.reason_to_id = {r: i for i, r in enumerate(REASONS)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = latent_tokens(sample, self.lang_latents, self.vis_latents)
        ids, mask = encode_tokens(tokens, self.vocab, MAX_LEN)
        lang_pos = torch.tensor([i for i, t in enumerate(tokens[:MAX_LEN]) if t.startswith("LANG_LATENT_")])
        vis_pos = torch.tensor([i for i, t in enumerate(tokens[:MAX_LEN]) if t.startswith("VIS_LATENT_")])
        return ids, mask, lang_pos, vis_pos, future_to_tensor(sample), torch.tensor(self.reason_to_id[reason_from_sample(sample)]), future_world_target(sample)


class OneVLStyleToy(nn.Module):
    def __init__(self, vocab_size, hidden_dim=96):
        super().__init__()
        self.token = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos = nn.Embedding(MAX_LEN, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, 4, hidden_dim * 4, batch_first=True, activation="gelu")
        self.backbone = nn.TransformerEncoder(layer, num_layers=2)
        self.traj_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 128), nn.GELU(), nn.Linear(128, 12))
        self.language_aux = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, len(REASONS)))
        self.world_aux = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, ids, mask, lang_pos, vis_pos, trace=False):
        bsz, seq_len = ids.shape
        pos = torch.arange(seq_len, device=ids.device).unsqueeze(0).expand(bsz, seq_len)
        hidden = self.backbone(self.token(ids) + self.pos(pos), src_key_padding_mask=~mask.bool())
        batch = torch.arange(bsz, device=ids.device).unsqueeze(1)
        lang_h = hidden[batch, lang_pos.to(ids.device)]
        vis_h = hidden[batch, vis_pos.to(ids.device)]
        traj = self.traj_head(torch.cat([lang_h, vis_h], dim=1).mean(dim=1))
        reason = self.language_aux(lang_h.mean(dim=1))
        world = self.world_aux(vis_h.mean(dim=1))
        if trace:
            print("hidden:", hidden.shape)
            print("language_latent_h:", lang_h.shape)
            print("visual_latent_h:", vis_h.shape)
            print("trajectory:", traj.shape)
            print("reason_logits:", reason.shape)
            print("world_pred:", world.shape)
        return traj, reason, world


def set_grad(module, value):
    for p in module.parameters():
        p.requires_grad_(value)


def configure_stage(model, stage):
    set_grad(model, True)
    if stage == 1:
        set_grad(model.token, False); set_grad(model.pos, False); set_grad(model.backbone, False); set_grad(model.traj_head, False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=10000)
    parser.add_argument("--stage0-epochs", type=int, default=3)
    parser.add_argument("--stage1-epochs", type=int, default=2)
    parser.add_argument("--stage2-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    samples = ensure_data(num_samples=args.num_samples)
    train, _ = split_samples(samples)
    extra = [f"LANG_LATENT_{i}" for i in range(4)] + [f"VIS_LATENT_{i}" for i in range(4)]
    vocab = build_vocab(train, extra_tokens=extra)
    loader = DataLoader(OneVLToyDataset(train, vocab, 4, 4), batch_size=args.batch_size, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = OneVLStyleToy(len(vocab)).to(device)
    first = next(iter(loader))
    print("source = OneVL: https://github.com/xiaomi-research/OneVL")
    model(first[0].to(device), first[1].to(device), first[2].to(device), first[3].to(device), trace=True)
    stages = [(0, args.stage0_epochs), (1, args.stage1_epochs), (2, args.stage2_epochs)]
    for stage, epochs in stages:
        configure_stage(model, stage)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-3)
        print(f"stage {stage}")
        for epoch in range(1, epochs + 1):
            total = 0.0
            count = 0
            for ids, mask, lp, vp, traj_y, reason_y, world_y in loader:
                traj, reason, world = model(ids.to(device), mask.to(device), lp.to(device), vp.to(device))
                traj_loss = F.mse_loss(traj, traj_y.to(device))
                lang_loss = F.cross_entropy(reason, reason_y.to(device))
                world_loss = F.mse_loss(world, world_y.to(device))
                if stage == 0:
                    loss = traj_loss
                elif stage == 1:
                    loss = lang_loss + world_loss
                else:
                    loss = traj_loss + 0.2 * (lang_loss + world_loss)
                opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * ids.shape[0]; count += ids.shape[0]
            print(f"epoch {epoch:02d} loss={total / count:.6f}")
    print("inference note: auxiliary decoders can be removed; trajectory still uses latent hidden states.")


if __name__ == "__main__":
    main()
