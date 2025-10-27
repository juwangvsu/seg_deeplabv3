
# utils_io.py
import os
import json
from typing import Optional

import numpy as np
from PIL import Image

import torch

from radar_dataset import apply_palette, CITYSCAPES_PALETTE

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_checkpoint(ckpt_dir: str, epoch: int, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    best: bool, best_metric: float):
    ensure_dir(ckpt_dir)
    ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch:04d}.pt')
    payload = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_metric': best_metric,
    }
    torch.save(payload, ckpt_path)
    if best:
        best_path = os.path.join(ckpt_dir, 'best.pt')
        torch.save(payload, best_path)
        meta = {
            'epoch': epoch,
            'best_metric': best_metric,
        }
        with open(os.path.join(ckpt_dir, 'best_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

def load_checkpoint(ckpt_dir: str, ckpt_name: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None) -> int:
    path = os.path.join(ckpt_dir, ckpt_name)
    payload = torch.load(path, map_location='cpu')
    model.load_state_dict(payload['model'])
    if optimizer is not None and 'optimizer' in payload and payload['optimizer']:
        optimizer.load_state_dict(payload['optimizer'])
    return int(payload.get('epoch', 0))

def save_pred_images(out_root: str, stem: str, pred_mask: np.ndarray,
                     rgb_path: Optional[str] = None):
    """
    Save raw mask, colored mask, and overlay (if rgb available) to:
      out_root/masks, out_root/colored_masks, out_root/overlay
    """
    masks_dir = os.path.join(out_root, 'masks')
    cdir = os.path.join(out_root, 'colored_masks')
    odir = os.path.join(out_root, 'overlay')
    ensure_dir(masks_dir); ensure_dir(cdir); ensure_dir(odir)

    # raw mask
    raw = Image.fromarray(pred_mask.astype(np.uint8), mode='L')
    raw.save(os.path.join(masks_dir, f'{stem}.png'))

    # colored
    color = apply_palette(pred_mask)
    color.save(os.path.join(cdir, f'{stem}.png'))

    # overlay
    if rgb_path and os.path.exists(rgb_path):
        rgb = Image.open(rgb_path).convert('RGB')
        # resize colored to match RGB for overlay
        color_rgba = color.convert('RGBA')
        rgb = rgb.resize(color.size, resample=Image.BILINEAR)
        overlay = Image.blend(rgb, color_rgba.convert('RGB'), alpha=0.5)
        overlay.save(os.path.join(odir, f'{stem}.png'))
