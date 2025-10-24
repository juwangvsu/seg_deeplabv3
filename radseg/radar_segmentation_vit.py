#!/usr/bin/env python3
"""
Radar Range–Angle Semantic Segmentation using a Transformer (ViT-style) backbone.

- Input: single-channel float32 range–angle maps stored as .npy (H, W) or .pt tensors.
- Labels: PNG files of class indices (same basename as input), e.g., foo.npy -> foo.png
- Backbone: Patch embedding (Conv2d) + ViT encoder (no CLS token)
- Decoder: Lightweight upsampling head to original resolution

CLI examples:

  # Train
  python radar_segmentation_vit.py \
    --data-dir ../data/ \
    --num-classes 19 \
    --epochs 50 \
    --batch-size 8 \
    --patch-size 16 \
    --img-size 512 512

  # Evaluate a checkpoint
  python radar_segmentation_vit.py --data-dir /path --eval --ckpt ckpt.pt --num-classes 6

Assumptions:
- Input files are *.npy or *.pt (float32 range–angle maps). Matching label PNGs share basename.
- Labels are uint8/uint16 PNGs of class IDs in [0..C-1]; use --ignore-index for unlabeled (e.g., 255).

"""
from __future__ import annotations
import argparse
import math
import os
from glob import glob
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# ----------------------
# Dataset
# ----------------------
class RadarSegDataset(Dataset):
    def __init__(self, data_dir: str, img_size: Tuple[int, int], normalize: str = "per_image_zscore",
                 inputs_subdir: str = "angle_range_numpy", masks_subdir: str = "masks",
                 label_ext: str = ".png"):
        self.data_dir = Path(data_dir)
        self.img_size = tuple(img_size)
        self.normalize = normalize
        self.inputs_subdir = inputs_subdir
        self.masks_subdir = masks_subdir
        self.label_ext = label_ext
        self.samples = self._discover()
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found with inputs in '{inputs_subdir}' and masks in '{masks_subdir}' under {data_dir}")

    def _discover(self) -> List[Tuple[Path, Path]]:
        inputs_root = self.data_dir / self.inputs_subdir
        masks_root = self.data_dir / self.masks_subdir
        files = sorted(glob(str(inputs_root / "**/*.npy"), recursive=True))
        pairs = []
        for f in files:
            p = Path(f)
            base = p.stem
            label = masks_root / f"{base}{self.label_ext}"
            if not label.exists():
                alt = masks_root / f"{base}.label{self.label_ext}"
                if alt.exists():
                    label = alt
                else:
                    continue
            pairs.append((p, label))
        return pairs

    def __len__(self):
        return len(self.samples)

    def _load_array(self, p: Path) -> np.ndarray:
        if p.suffix == ".npy":
            arr = np.load(p).astype(np.float32)
        else:
            raise ValueError(f"Unsupported input suffix: {p.suffix}; expected .npy under {self.inputs_subdir}")
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D range–angle map, got shape {arr.shape} for {p}")
        return arr

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self.normalize == "per_image_zscore":
            m, s = x.mean(), x.std()
            s = 1e-6 if s == 0 else s
            x = (x - m) / s
        elif self.normalize == "minmax":
            mn, mx = x.min(), x.max()
            denom = max(mx - mn, 1e-6)
            x = (x - mn) / denom
        elif self.normalize == "log1p":
            x = np.log1p(np.maximum(x, 0.0))
            m, s = x.mean(), x.std()
            s = 1e-6 if s == 0 else s
            x = (x - m) / s
        else:
            pass
        return x.astype(np.float32)

    def __getitem__(self, idx: int):
        ipath, lpath = self.samples[idx]
        x = self._load_array(ipath)
        x = self._normalize(x)
        H, W = x.shape
        target_h, target_w = self.img_size
        x_t = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        x_t = F.interpolate(x_t, size=(target_h, target_w), mode="bilinear", align_corners=False)
        x_t = x_t.squeeze(0)  # [1,H,W]

        y = Image.open(lpath).convert("I")  # 32-bit int pixels
        y = np.array(y)
        y_t = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float()
        y_t = F.interpolate(y_t, size=(target_h, target_w), mode=("nearest")).squeeze(0).squeeze(0).long()

        return {"image": x_t, "mask": y_t, "path": str(ipath)}

# ----------------------
# Vision Transformer Backbone (minimal)
# ----------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size: Tuple[int, int], patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_h = img_size[0] // patch_size
        self.grid_w = img_size[1] // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, self.grid_h * self.grid_w, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: [B, 1, H, W]
        x = self.proj(x)  # [B, C, H/ps, W/ps]
        B, C, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = x + self.pos[:, : x.size(1), :]
        return x, h, w


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x, need_weights=False)
        x = self.drop_path(x)
        x = x + h
        h = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + h
        return x


class ViTEncoder(nn.Module):
    def __init__(self, img_size: Tuple[int, int], patch_size: int = 16, in_chans: int = 1,
                 embed_dim: int = 256, depth: int = 8, num_heads: int = 8, mlp_ratio: float = 4.0,
                 drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        assert img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0, \
            "img_size must be divisible by patch_size"
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop, attn_drop) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        x, h, w = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, h, w  # [B, N, C], h, w


# ----------------------
# Decoder Head
# ----------------------
class SimpleDecoder(nn.Module):
    """Reshape tokens back to feature map and upsample to logits."""
    def __init__(self, num_classes: int, embed_dim: int, up_factor: int):
        super().__init__()
        # two-stage upsampling with convs
        mid = embed_dim // 2
        self.conv1 = nn.Conv2d(embed_dim, mid, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, mid, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(mid)
        self.classifier = nn.Conv2d(mid, num_classes, kernel_size=1)
        self.up_factor = up_factor

    def forward(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # tokens: [B, N, C]
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, h, w)  # [B,C,h,w]
        # upsample to original size
        x = F.interpolate(x, scale_factor=self.up_factor, mode="bilinear", align_corners=False)
        x = self.bn1(self.conv1(x))
        x = F.gelu(x)
        x = self.bn2(self.conv2(x))
        x = F.gelu(x)
        logits = self.classifier(x)  # [B, num_classes, H, W]
        return logits


class RadarViTSeg(nn.Module):
    def __init__(self, img_size: Tuple[int, int], num_classes: int, patch_size: int = 16,
                 embed_dim: int = 256, depth: int = 8, num_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.encoder = ViTEncoder(img_size, patch_size, 1, embed_dim, depth, num_heads, mlp_ratio)
        up_factor = patch_size
        self.decoder = SimpleDecoder(num_classes, embed_dim, up_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, h, w = self.encoder(x)
        logits = self.decoder(tokens, h, w)
        return logits


# ----------------------
# Metrics (mIoU, pixel acc)
# ----------------------
@torch.no_grad()
def compute_confusion_matrix(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int) -> torch.Tensor:
    # pred, target: [B,H,W] (integers)
    mask = target != ignore_index
    pred = pred[mask].view(-1)
    tgt = target[mask].view(-1)
    k = (tgt >= 0) & (tgt < num_classes)
    pred = pred[k]
    tgt = tgt[k]
    cm = torch.bincount(num_classes * tgt + pred, minlength=num_classes**2)
    cm = cm.reshape(num_classes, num_classes).to(torch.int64)
    return cm


def miou_from_confmat(cm: torch.Tensor) -> Tuple[float, List[float]]:
    # cm: [C,C] (rows = gt, cols = pred)
    tp = cm.diag().float()
    fp = cm.sum(0).float() - tp
    fn = cm.sum(1).float() - tp
    denom = tp + fp + fn
    ious = torch.where(denom > 0, tp / denom.clamp_min(1e-6), torch.zeros_like(denom))
    miou = ious.mean().item()
    return miou, ious.tolist()


@torch.no_grad()
def pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int) -> float:
    mask = target != ignore_index
    correct = (pred[mask] == target[mask]).sum().item()
    total = mask.sum().item()
    return float(correct) / max(total, 1)


# ----------------------
# Training / Eval
# ----------------------

def train_one_epoch(model, loader, optimizer, scaler, device, num_classes, ignore_index):
    model.train()
    total_loss = 0.0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(x)
            loss = F.cross_entropy(logits, y, ignore_index=ignore_index)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index):
    model.eval()
    cm_total = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    acc_total = 0.0
    n = 0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["mask"].to(device)
        logits = model(x)
        pred = logits.argmax(1)
        cm = compute_confusion_matrix(pred.cpu(), y.cpu(), num_classes, ignore_index)
        cm_total += cm
        acc_total += pixel_accuracy(pred, y, ignore_index) * x.size(0)
        n += x.size(0)
    miou, _ = miou_from_confmat(cm_total)
    return acc_total / max(n, 1), miou


# ----------------------
# CLI
# ----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Radar Range–Angle Segmentation with ViT Backbone")
    p.add_argument("--data-dir", required=True, help="Root directory that contains subfolders: angle_range_numpy/ and masks/")
    p.add_argument("--inputs-subdir", type=str, default="angle_range_numpy", help="Subfolder for .npy range–angle maps")
    p.add_argument("--masks-subdir", type=str, default="masks", help="Subfolder for mask PNGs")
    p.add_argument("--img-size", nargs=2, type=int, default=[512, 512], help="Resize (H W)")
    p.add_argument("--num-classes", type=int, required=True)
    p.add_argument("--ignore-index", type=int, default=255)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--split", type=float, default=0.9, help="Train split fraction")
    p.add_argument("--amp", action="store_true", help="Use mixed precision")
    p.add_argument("--ckpt", type=str, default="ckpt.pt")
    p.add_argument("--eval", action="store_true")
    return p.parse_args()


def make_loaders(args):
    ds = RadarSegDataset(
        args.data_dir,
        img_size=tuple(args.img_size),
        inputs_subdir=args.inputs_subdir,
        masks_subdir=args.masks_subdir,
    )
    n_train = int(len(ds) * args.split)
    n_val = len(ds) - n_train
    train_set, val_set = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RadarViTSeg(
        img_size=tuple(args.img_size),
        num_classes=args.num_classes,
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
    ).to(device)

    if args.eval and os.path.exists(args.ckpt):
        model.load_state_dict(torch.load(args.ckpt, map_location=device))

    train_loader, val_loader = make_loaders(args)

    if args.eval:
        acc, miou = evaluate(model, val_loader, device, args.num_classes, args.ignore_index)
        print(f"Eval — pixel_acc: {acc:.4f}, mIoU: {miou:.4f}")
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if (args.amp and device.type == 'cuda') else None

    best_miou = -1.0
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, args.num_classes, args.ignore_index)
        acc, miou = evaluate(model, val_loader, device, args.num_classes, args.ignore_index)
        print(f"Epoch {epoch:03d}: loss={loss:.4f} acc={acc:.4f} miou={miou:.4f}")
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), args.ckpt)
            print(f"Saved checkpoint to {args.ckpt} (best mIoU {best_miou:.4f})")


if __name__ == "__main__":
    main()

