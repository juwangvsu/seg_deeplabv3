# train.py  (minimal, readable skeleton)
import os, math, time, argparse
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from pathlib import Path
import random
from radcam.model import RadarCameraSeg, build_overlap_masks  # from previous pseudocode
from utils.train_utils import load_config
from radcam.eval import run_eval
from radcam.dataset import RadarCamSegDataset as RadarCamSegDataset 
# ----------------------------
# Dataset (replace with your loaders)
# ----------------------------
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from tqdm import tqdm
def sanitize(x):
    # replace NaN/Inf with 0
    x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))
    return x

# Example transforms (adapt to your data)
def default_transforms(HW=(512,896), RA=(256,256)):
    img_tfms = T.Compose([
        T.ToTensor(),                              # [0,1]
        T.Resize(HW, antialias=True),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    # radar is already a tensor [1,R,A]; just resize to RA if needed
    def radar_tfms(x):  # x: [1,R,A]
        x = x.unsqueeze(0) if x.ndim==3 else x  # [1,1,R,A]
        x = torch.nn.functional.interpolate(x, size=RA, mode="bilinear", align_corners=False)
        return x[0]
    return img_tfms, radar_tfms

# ----------------------------
# Losses
# ----------------------------
class SafeCrossEntropy2D(torch.nn.Module):
    def __init__(self, ignore_index=255, label_smoothing=0.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.ls = label_smoothing

    def forward(self, logits, target):
        # logits: [B,C,H,W], target: [B,H,W] (torch.long)
        B, C, H, W = logits.shape
        assert target.dtype == torch.long, f"target dtype must be long, got {target.dtype}"
        # per-pixel loss
        loss = torch.nn.functional.cross_entropy(
            logits, target,
            ignore_index=self.ignore_index,
            reduction='none',
            label_smoothing=self.ls
        )  # [B,H,W]
        valid = (target != self.ignore_index)  # [B,H,W]
        valid_count = valid.sum().clamp_min(1)
        loss = (loss * valid).sum() / valid_count
        return loss

class Losses(nn.Module):
    def __init__(self, num_classes, ignore_index=255, aux_lambda=0.3):
        super().__init__()
        #self.seg = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.seg = SafeCrossEntropy2D(ignore_index=ignore_index)  # <—
        self.aux_lambda = aux_lambda
        self.bce = nn.BCEWithLogitsLoss(reduction='none')  # we'll mask this too

    def forward(self, seg_logits, seg_target, aux_logits=None, aux_target=None, aux_mask=None):
        # seg_logits: [B,K,H,W]; seg_target: [B,H,W]
        loss = self.seg(seg_logits, seg_target)

        print('zzz loss crossentropy logits shape, target shape', loss, seg_logits.shape, seg_logits.dtype, seg_target.shape, seg_target.dtype, seg_target[0])

        if aux_logits is not None and aux_target is not None:
            # aux_logits: [B,Nr,2] -> treat as 2 independent binary targets (e.g., obj and vel>0)
            aux = self.bce(aux_logits, aux_target)  # [B,Nr,2]
            if aux_mask is not None:
                aux = aux * aux_mask  # mask invalid radar cells if needed
                denom = aux_mask.sum().clamp_min(1)
                aux = aux.sum() / denom
            else:
                aux = aux.mean()
            loss = loss + self.aux_lambda * aux
        return loss

# ----------------------------
# Utility: modality dropout
# ----------------------------
def modality_dropout(img, radar, p_img=0.0, p_radar=0.1):
    if p_img>0 and random.random()<p_img:
        img = torch.zeros_like(img)
    if p_radar>0 and random.random()<p_radar:
        radar = torch.zeros_like(radar)
    return img, radar

def assert_finite(t, name):
    if not torch.isfinite(t).all():
        mn = torch.nanmin(t).item() if torch.isnan(t).any() else t.min().item()
        mx = torch.nanmax(t).item() if torch.isnan(t).any() else t.max().item()
        raise RuntimeError(f"{name} has non-finite values (min={mn}, max={mx})")

# ----------------------------
# Train / Validate loops
# ----------------------------
def train_epoch(model, loader, optim, scaler, losses, device, calib=None, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", unit="batch")

    for batch in pbar:

        img   = sanitize(batch["image"].to(device, non_blocking=True).float())
        radar = sanitize(batch["radar"].to(device, non_blocking=True).float())
        target= batch["mask"].to(device, non_blocking=True).long()

        # (optional) modality dropout to ensure radar is used
        img, radar = modality_dropout(img, radar)
        use_amp=False
        with torch.cuda.amp.autocast(enabled=use_amp):
            # Build sparsity masks (optional, expensive): set to None to skip
            overlap = None
            if calib is not None:
                # You may cache Hc,Wc,Hr,Wr after first forward to avoid recompute
                pass
            seg_logits, aux_logits = model(img, radar, overlap_masks=overlap)
            assert_finite(seg_logits, "seg_logits")

            # Optional auxiliary targets for radar; here we create dummy zeros:
            aux_target = None
            if aux_logits is not None:
                aux_target = torch.zeros_like(aux_logits)  # shape [B,Nr,2]

            loss = losses(seg_logits, target, aux_logits, aux_target)

        optim.zero_grad(set_to_none=True)
        if scaler is None:
            loss.backward()
            if clip_grad: nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optim.step()
        else:
            scaler.scale(loss).backward()
            if clip_grad: scaler.unscale_(optim); nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optim); scaler.update()

        total_loss += loss.item()
        avg_loss = total_loss / (pbar.n + 1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{avg_loss:.4f}"})

    pbar.close()
    return total_loss / max(1,len(loader))

@torch.no_grad()
def validate(model, loader, losses, device):
    model.eval()
    total = 0.0
    for batch in loader:
        img   = batch["image"].to(device)
        radar = batch["radar"].to(device)
        target= batch["mask"].to(device)
        seg_logits, aux_logits = model(img, radar, overlap_masks=None)
        loss = losses(seg_logits, target, aux_logits=None, aux_target=None)
        total += loss.item()
    return total / max(1,len(loader))

# ----------------------------
# Main
# ----------------------------
def main(num_classes=19, epochs=50, lr=2e-4, wd=0.01, amp=True, device="cuda"):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument('--eval', action='store_true',
                   help='Run inference on val_images/val_radar and save masks/overlays.')
    ap.add_argument('--load', type=str, default=None,
                   help='Path to checkpoint (.pt) with model state_dict.')
    args = ap.parse_args()

    cfg = load_config(args.config)
    dcfg = cfg["data"]
    root = dcfg["root"]

    out_dir = cfg["train"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 1337))

    train_set = RadarCamSegDataset(
        image_dir=os.path.join(root, dcfg["train_images"]),
        mask_dir=os.path.join(root, dcfg["train_masks"]),
        radar_dir=os.path.join(root, dcfg["radar_npy"]),
        num_classes=19,
        augment=True
    )

    val_set = RadarCamSegDataset(
        image_dir=os.path.join(root, dcfg["train_images"]),
        mask_dir=os.path.join(root, dcfg["train_masks"]),
        radar_dir=os.path.join(root, dcfg["radar_npy"]),
        num_classes=19
    )

    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)


    model = RadarCameraSeg(num_classes=num_classes, d=256, heads=8).to(device)

    # (optional) load
    if args.load:
        load_checkpoint(model, args.load, device=device)

    # EVAL mode only
    if args.eval:
        run_eval(
            model=model,
            image_dir=os.path.join(root, dcfg["val_images"]),
            mask_dir=os.path.join(root, dcfg["train_masks"]),
            radar_dir=os.path.join(root, dcfg["radar_npy"]),
            #radar_dir=dcfg["radar_npy"],
            out_dir=out_dir
        )
        return

    ######## below training logic #####################

    with torch.no_grad():
        img   = torch.zeros(2,3,512,896, device=device)
        radar = torch.zeros(2,1,256,256, device=device)
        seg_logits, _ = model(img, radar)
        tgt = torch.zeros(2,512,896, dtype=torch.long, device=device)
        test_loss = SafeCrossEntropy2D()(seg_logits, tgt)
        print("synthetic test loss:", test_loss.item())  # should be finite
    #exit(0)
    # Optimizer / schedule
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    losses = Losses(num_classes=num_classes, ignore_index=255, aux_lambda=0.3)

    best_val = float("inf")
    for ep in range(1, epochs+1):
        tr = train_epoch(model, train_loader, optim, scaler, losses, device, calib=None, clip_grad=1.0)
        va = validate(model, val_loader, losses, device)
        sched.step()

        print(f"Epoch {ep:03d} | train {tr:.4f} | val {va:.4f} | lr {optim.param_groups[0]['lr']:.2e}")

        if va < best_val:
            best_val = va
            torch.save({"ep": ep, "model": model.state_dict()}, "best_model.pt")

    # final checkpoint
    torch.save({"ep": epochs, "model": model.state_dict()}, "last_model.pt")

# -------------
# Usage sketch
# -------------
# train_samples / val_samples should be lists of dicts:
# train_samples = [
#   {"image": PIL_or_tensor_img, "radar": torch.tensor([1,R,A]), "mask": torch.tensor([H,W], dtype=torch.long)},
#   ...
# ]

def load_checkpoint(model, ckpt_path, device='cpu'):
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model', ckpt)  # support either {'model': ...} or plain SD
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] loaded from {ckpt_path}")
    if missing:    print("[load] missing keys:", missing)
    if unexpected: print("[load] unexpected keys:", unexpected)

if __name__ == "__main__":
    main()

#main(train_samples, val_samples, num_classes=K)

