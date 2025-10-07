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
# ----------------------------
# Dataset (replace with your loaders)
# ----------------------------
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from tqdm import tqdm

class RadarCamSegDataset(Dataset):
    """
    Loads triplets (image, radar, mask) for radar+camera semantic segmentation.

    Each triplet shares the same filename stem:
        image_dir/000123.jpg
        mask_dir/000123.png
        radar_dir/000123.npy   (float32 range–angle map)

    Args:
        image_dir (str|Path): path to RGB images (jpg/png)
        mask_dir  (str|Path): path to segmentation masks (png)
        radar_dir (str|Path): path to radar range–angle npy files
        num_classes (int): number of semantic classes
        img_size (tuple): target (H, W) for image/mask resize
        radar_size (tuple): target (R, A) for radar resize
        ignore_index (int): label value to ignore in loss
        augment (bool): whether to apply random flips/crops
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        radar_dir,
        num_classes,
        img_size=(512, 896),
        radar_size=(256, 256),
        ignore_index=255,
        augment=False,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.radar_dir = Path(radar_dir)
        self.num_classes = num_classes
        self.img_size = img_size
        self.radar_size = radar_size
        self.ignore_index = ignore_index
        self.augment = augment

        # Match files by stem (excluding extension)
        self.samples = []
        img_files = sorted(self.image_dir.glob("*"))
        for img_path in img_files:
            stem = img_path.stem
            mask_path = self.mask_dir / f"{stem}.png"
            radar_path = self.radar_dir / f"{stem}.npy"
            if mask_path.exists() and radar_path.exists():
                self.samples.append((img_path, radar_path, mask_path))

        if not self.samples:
            raise RuntimeError(f"No matching triplets found in {image_dir}, {mask_dir}, {radar_dir}")

        # --- Transforms ---
        self.img_tfms = T.Compose([
            T.Resize(img_size, antialias=True),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

        # Masks → tensor without normalization
        self.mask_tfms = T.Compose([T.Resize(img_size, interpolation=T.InterpolationMode.NEAREST)])

    def __len__(self):
        return len(self.samples)

    def _load_radar(self, path):
        """Load radar range–angle numpy array [R,A] and resize to target."""
        arr = np.load(path)  # float32, shape [R,A] or [1,R,A]
        if arr.ndim == 2:
            arr = arr[None]  # [1,R,A]
        tensor = torch.from_numpy(arr).float().unsqueeze(0)  # [1,1,R,A]
        tensor = torch.nn.functional.interpolate(
            tensor, size=self.radar_size, mode="bilinear", align_corners=False
        )[0]  # [1,R',A']
        return tensor

    def __getitem__(self, idx):
        img_path, radar_path, mask_path = self.samples[idx]

        # --- Load image ---
        img = Image.open(img_path).convert("RGB")
        img = self.img_tfms(img)  # [3,H,W]

        # --- Load radar ---
        radar = self._load_radar(radar_path)  # [1,R,A]

        # --- Load mask ---
        mask = Image.open(mask_path)
        mask = self.mask_tfms(mask)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))  # [H,W]

        # --- Optional augmentations ---
        if self.augment and random.random() < 0.5:
            img = torch.flip(img, dims=[2])
            radar = torch.flip(radar, dims=[2])
            mask = torch.flip(mask, dims=[1])

        return {"image": img, "radar": radar, "mask": mask}

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
class Losses(nn.Module):
    def __init__(self, num_classes, ignore_index=255, aux_lambda=0.3):
        super().__init__()
        self.seg = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.aux_lambda = aux_lambda
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, seg_logits, seg_target, aux_logits=None, aux_target=None):
        # seg_logits: [B,K,H,W]; seg_target: [B,H,W]
        loss = self.seg(seg_logits, seg_target)
        if aux_logits is not None and aux_target is not None:
            # aux_logits: [B,Nr,2] -> treat as 2 independent binary targets (e.g., obj and vel>0)
            loss = loss + self.aux_lambda * self.bce(aux_logits, aux_target)
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

# ----------------------------
# Train / Validate loops
# ----------------------------
def train_epoch(model, loader, optim, scaler, losses, device, calib=None, clip_grad=1.0):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="Training", unit="batch")

    for batch in pbar:
        img   = batch["image"].to(device, non_blocking=True)   # [B,3,H,W]
        radar = batch["radar"].to(device, non_blocking=True)   # [B,1,R,A]
        target= batch["mask"].to(device, non_blocking=True)    # [B,H,W]

        # (optional) modality dropout to ensure radar is used
        img, radar = modality_dropout(img, radar)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            # Build sparsity masks (optional, expensive): set to None to skip
            overlap = None
            if calib is not None:
                # You may cache Hc,Wc,Hr,Wr after first forward to avoid recompute
                pass
            seg_logits, aux_logits = model(img, radar, overlap_masks=overlap)

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
if __name__ == "__main__":
    main()

#main(train_samples, val_samples, num_classes=K)

