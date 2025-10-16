#!/usr/bin/env python3
"""
train_finetune_segformer.py

Fine-tune Hugging Face Transformers' SegFormer on Cityscapes-style data using a
**custom PyTorch training loop** (no Transformers Trainer).

Features
- Cityscapes pairing (leftImg8bit ↔ gtFine), labelIds→trainIds mapping (optional)
- Mixed precision (fp16/bf16) with GradScaler
- Gradient accumulation, cosine decay with linear warmup
- Pixel Accuracy & mIoU (ignore_index=255) on the eval set each epoch
- Checkpointing: best (by mIoU), last, and per-epoch
- Resume training (model+optimizer+scheduler+scaler)

Example:

python train_finetune_segformer.py \
  --model-id nvidia/segformer-b2-finetuned-cityscapes-1024-1024 \
  --train-input-dir /data/cityscapes/leftImg8bit/train \
  --train-mask-dir  /data/cityscapes/gtFine/train \
  --val-input-dir   /data/cityscapes/leftImg8bit/val \
  --val-mask-dir    /data/cityscapes/gtFine/val \
  --output-dir      ./segformer_city_ft \
  --batch-size 2 --lr 6e-5 --epochs 30 --img-height 512 --img-width 1024 --fp16

"""
from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# ============================
# Cityscapes constants & utils
# ============================
CITYSCAPES_CLASSES = 19
CITYSCAPES_IGNORE_INDEX = 255

CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole",
    "traffic light", "traffic sign", "vegetation", "terrain", "sky",
    "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"
]

# labelId -> trainId mapping (vectorized). Unspecified ids map to 255.
_LABELID_TO_TRAINID = {
    0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255,
    7:0, 8:1, 9:255, 10:255,
    11:2, 12:3, 13:4, 14:255, 15:255, 16:255,
    17:5, 18:255, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15,
    29:255, 30:255, 31:16, 32:17, 33:18,
    -1:255  # sometimes unlabeled is -1
}

def map_labelIds_to_trainIds(arr: np.ndarray) -> np.ndarray:
    out = np.full_like(arr, CITYSCAPES_IGNORE_INDEX)
    for lid, tid in _LABELID_TO_TRAINID.items():
        out[arr == lid] = tid
    return out.astype(np.uint8)

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

def list_cityscapes_images(input_dir: Path) -> List[Path]:
    imgs = sorted(input_dir.rglob("*_leftImg8bit.png"))
    if not imgs:
        imgs = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not imgs:
        raise FileNotFoundError(f"No images found under {input_dir}")
    return imgs


def match_cityscapes_mask(img_path: Path, input_dir: Path, mask_dir: Path) -> Optional[Path]:
    """Match a leftImg8bit image to its gtFine mask (trainIds preferred)."""
    rel = img_path.relative_to(input_dir)
    stem = img_path.stem.replace("_leftImg8bit", "")

    if len(rel.parts) >= 3:
        split, city = rel.parts[-3], rel.parts[-2]
        c1 = mask_dir / split / city / f"{stem}_gtFine_labelTrainIds.png"
        c2 = mask_dir / split / city / f"{stem}_gtFine_labelIds.png"
        if c1.exists():
            return c1
        if c2.exists():
            return c2

    # Fallback search
    c1_list = list(mask_dir.rglob(f"{stem}_gtFine_labelTrainIds.png"))
    if c1_list:
        return c1_list[0]
    c2_list = list(mask_dir.rglob(f"{stem}_gtFine_labelIds.png"))
    if c2_list:
        return c2_list[0]
    return None


# ============================
# Dataset & Collator
# ============================
class CityscapesSegDataset(Dataset):
    def __init__(self, input_dir: Path, mask_dir: Path, assume_trainIds: bool = True):
        self.input_dir = Path(input_dir)
        self.mask_dir = Path(mask_dir)
        self.assume_trainIds = assume_trainIds
        self.images = list_cityscapes_images(self.input_dir)
        self.pairs: List[Tuple[Path, Path]] = []
        skipped = 0
        for p in self.images:
            m = match_cityscapes_mask(p, self.input_dir, self.mask_dir)
            if m is not None:
                self.pairs.append((p, m))
            else:
                skipped += 1
        if not self.pairs:
            raise RuntimeError("No image/mask pairs matched. Check your directories.")
        if skipped:
            print(f"[warn] Skipped {skipped} images without matching GT mask")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict:
        ip, mp = self.pairs[idx]
        img = Image.open(ip).convert("RGB")
        mask_np = np.array(Image.open(mp))
        if not self.assume_trainIds:
            mask_np = map_labelIds_to_trainIds(mask_np)
        mask_np = mask_np.astype(np.uint8)
        return {"image": img, "mask": mask_np}


@dataclass
class SegDataCollator:
    processor: AutoImageProcessor
    height: int
    width: int

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        images = [b["image"] for b in batch]
        masks  = [b["mask"]  for b in batch]
        enc = self.processor(
            images=images,
            segmentation_maps=masks,
            return_tensors="pt",
            do_resize=True,
            size={"height": self.height, "width": self.width},
        )
        # enc => {pixel_values: [B,3,H,W], labels: [B,H,W]}
        return enc


# ============================
# Metrics (acc, mIoU)
# ============================

def fast_hist(pred: np.ndarray, tgt: np.ndarray, n_class: int, ignore_index: int) -> np.ndarray:
    print('xxx tgt.shape pred.shape ', tgt.shape, pred.shape, n_class)
    mask =( tgt != ignore_index) & (tgt>=0) & (tgt < n_class)
    if mask.sum() == 0:
        return np.zeros((n_class, n_class), dtype=np.int64)
    hist = np.bincount(
        n_class * tgt[mask].astype(np.int64) + pred[mask].astype(np.int64),
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def compute_from_hist(hist: np.ndarray) -> Tuple[float, float, np.ndarray]:
    acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
    valid = ~np.isnan(iu)
    miou = iu[valid].mean() if valid.any() else float("nan")
    return float(acc), float(miou), iu


# ============================
# LR Scheduler (warmup + cosine)
# ============================

def build_warmup_cosine(optimizer: torch.optim.Optimizer, num_warmup_steps: int, num_training_steps: int):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def _weights_init(m: nn.Module):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm1d, nn.BatchNorm2d)):
        if getattr(m, 'weight', None) is not None:
            nn.init.ones_(m.weight)
        if getattr(m, 'bias', None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

def reinit_backbone(segformer_model: SegformerForSemanticSegmentation):
    if hasattr(segformer_model, 'segformer'):
        print('[reinit] Randomly initializing backbone (segformer) weights')
        segformer_model.segformer.apply(_weights_init)
    else:
        print('[reinit] WARNING: Model has no attribute `segformer` — skip backbone reinit')


def reinit_decode_head(segformer_model: SegformerForSemanticSegmentation):
    if hasattr(segformer_model, 'decode_head'):
        print('[reinit] Randomly initializing decode head weights')
        segformer_model.decode_head.apply(_weights_init)
        # Ensure classifier conv bias exists & is zeroed (if present)
        classifier = getattr(segformer_model.decode_head, 'classifier', None)
        if isinstance(classifier, nn.Conv2d) and classifier.bias is not None:
            nn.init.zeros_(classifier.bias)
    else:
        print('[reinit] WARNING: Model has no attribute `decode_head` — skip head reinit')


# ============================
# Utils
# ============================

def save_checkpoint(state: Dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, model: torch.nn.Module, optimizer=None, scheduler=None, scaler=None):
    chk = torch.load(path, map_location="cpu")
    model.load_state_dict(chk["model_state"])
    if optimizer is not None and "optimizer_state" in chk:
        optimizer.load_state_dict(chk["optimizer_state"])
    if scheduler is not None and "scheduler_state" in chk and scheduler is not None:
        scheduler.load_state_dict(chk["scheduler_state"])
    if scaler is not None and "scaler_state" in chk:
        scaler.load_state_dict(chk["scaler_state"])
    start_epoch = chk.get("epoch", 0) + 1
    global_step = chk.get("global_step", 0)
    best_miou = chk.get("best_miou", float("nan"))
    return start_epoch, global_step, best_miou


# ============================
# Main Training Loop
# ============================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune SegFormer on Cityscapes (custom loop)")
    parser.add_argument("--model-id", type=str, required=True,
                        help="Base checkpoint (hub id or local path). Can be backbone or a prior finetune.")
    parser.add_argument("--train-input-dir", type=str, required=True, help="Cityscapes leftImg8bit/train dir")
    parser.add_argument("--train-mask-dir", type=str, required=True, help="Cityscapes gtFine/train dir")
    parser.add_argument("--val-input-dir", type=str, required=True, help="Cityscapes leftImg8bit/val dir")
    parser.add_argument("--val-mask-dir", type=str, required=True, help="Cityscapes gtFine/val dir")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save model + logs")

    parser.add_argument("--img-height", type=int, default=512)
    parser.add_argument("--img-width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint path (*.pth)")
    parser.add_argument("--ignore-mismatched", action="store_true", help="Ignore size mismatch when loading")

    parser.add_argument("--assume-trainIds", action="store_true",
                        help="Assume GT masks are *_labelTrainIds.png (skip labelIds->trainIds mapping)")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--rand-backbone", action="store_true", help="Randomly re-initialize backbone (segformer)")
    parser.add_argument("--rand-decode-head", action="store_true", help="Randomly re-initialize decode head")
    parser.add_argument("--rand-all", action="store_true", help="Randomly re-initialize both backbone and decode head")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Processor & model
    processor = AutoImageProcessor.from_pretrained(args.model_id)

    id2label = {i: name for i, name in enumerate(CITYSCAPES_CLASS_NAMES)}
    label2id = {v: k for k, v in id2label.items()}

    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_id,
        num_labels=CITYSCAPES_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=args.ignore_mismatched,
    )
    # Ensure loss ignores 255
    if hasattr(model.config, "ignore_index"):
        model.config.ignore_index = CITYSCAPES_IGNORE_INDEX
    if hasattr(model.config, "semantic_loss_ignore_index"):
        model.config.semantic_loss_ignore_index = CITYSCAPES_IGNORE_INDEX

    # Apply random re-initialization options
    if args.rand_all or args.rand_backbone:
        reinit_backbone(model)
    if args.rand_all or args.rand_decode_head:
        reinit_decode_head(model)

    model.to(device)

    # Datasets & loaders
    train_ds = CityscapesSegDataset(Path(args.train_input_dir), Path(args.train_mask_dir),
                                    assume_trainIds=args.assume_trainIds)
    val_ds   = CityscapesSegDataset(Path(args.val_input_dir),   Path(args.val_mask_dir),
                                    assume_trainIds=args.assume_trainIds)

    collator = SegDataCollator(processor=processor, height=args.img_height, width=args.img_width)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=max(1, args.batch_size//2), shuffle=False,
                              num_workers=4, pin_memory=True, collate_fn=collator)

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader))
    total_update_steps = (steps_per_epoch * args.epochs) // max(1, args.grad_accum)
    warmup_steps = int(args.warmup_ratio * total_update_steps)
    scheduler = build_warmup_cosine(optimizer, warmup_steps, total_update_steps)

    # AMP
    use_amp = bool(args.fp16 or args.bf16)
    amp_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)  # GradScaler only for fp16

    # Resume
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    global_step = 0
    best_miou = -1.0

    if args.resume_from:
        ckpt_path = Path(args.resume_from)
        if ckpt_path.exists():
            print(f"[resume] Loading checkpoint from {ckpt_path}")
            start_epoch, global_step, best_miou = load_checkpoint(
                ckpt_path, model, optimizer, scheduler, scaler
            )
        else:
            print(f"[warn] --resume-from {ckpt_path} not found; starting fresh")

    # --------------------
    # Training epochs
    # --------------------
    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(pixel_values=pixel_values, labels=labels)
                loss = out.loss / max(1, args.grad_accum)

            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (step + 1) % max(1, args.grad_accum) == 0:
                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1

            running += loss.item() * max(1, args.grad_accum)
            if (step + 1) % 20 == 0:
                avg = running / 20.0
                print(f"Epoch {epoch+1}/{args.epochs} | step {step+1}/{steps_per_epoch} | loss {avg:.4f}")
                running = 0.0

        # --------------------
        # Validation
        # --------------------
        model.eval()
        hist = np.zeros((CITYSCAPES_CLASSES, CITYSCAPES_CLASSES), dtype=np.int64)
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    out = model(pixel_values=pixel_values, labels=labels)
                    logits = out.logits  # [B,C,H,W] (same H,W as labels due to processor resize)
                    if logits.shape[-2:] != labels.shape[-2:]:
                        logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

                val_loss += out.loss.item()
                preds = logits.argmax(dim=1).detach().cpu().numpy().astype(np.uint8)
                gts = labels.detach().cpu().numpy().astype(np.uint8)

                for p, t in zip(preds, gts):
                    hist += fast_hist(p, t, CITYSCAPES_CLASSES, CITYSCAPES_IGNORE_INDEX)

        acc, miou, _ = compute_from_hist(hist)
        val_loss /= max(1, len(val_loader))
        print(f"[val] epoch {epoch+1}: loss={val_loss:.4f}, acc={acc*100:.2f}%, mIoU={miou*100:.2f}%")

        # --------------------
        # Checkpointing
        # --------------------
        # Save 'last'
        last_ckpt = output_dir / "last.pth"
        save_checkpoint({
            "epoch": epoch,
            "global_step": global_step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict() if args.fp16 else None,
            "best_miou": best_miou,
        }, last_ckpt)

        # Save per-epoch
        if (epoch + 1) % max(1, args.save_every) == 0:
            epoch_ckpt = output_dir / f"epoch_{epoch+1:03d}.pth"
            save_checkpoint({
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if args.fp16 else None,
                "best_miou": best_miou,
            }, epoch_ckpt)

        # Save best (by mIoU)
        if miou > best_miou:
            best_miou = miou
            print(f"[best] New best mIoU: {best_miou*100:.2f}% — saving")
            model.save_pretrained(output_dir / "best_model")
            processor.save_pretrained(output_dir / "best_model")
            save_checkpoint({
                "epoch": epoch,
                "global_step": global_step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "scaler_state": scaler.state_dict() if args.fp16 else None,
                "best_miou": best_miou,
            }, output_dir / "best.pth")

    # Save final
    model.save_pretrained(output_dir / "final_model")
    processor.save_pretrained(output_dir / "final_model")
    print("Training complete.")


if __name__ == "__main__":
    main()

