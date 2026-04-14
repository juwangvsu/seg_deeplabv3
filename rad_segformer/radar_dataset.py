
# radar_dataset.py
import os
import glob
from typing import Tuple, List, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

# ----------------------
# Simple Cityscapes-like palette (0..18). Extend if needed.
# If your dataset has a different class mapping, adjust here.
# ----------------------
CITYSCAPES_PALETTE = [
    128, 64,128,  # 0 road / bg
    244, 35,232,  # 1
     70, 70, 70,  # 2
    102,102,156,  # 3
    190,153,153,  # 4
    153,153,153,  # 5
    250,170, 30,  # 6
    220,220,  0,  # 7
    107,142, 35,  # 8
    152,251,152,  # 9
     70,130,180,  #10
    220, 20, 60,  #11
    255,  0,  0,  #12
      0,  0,142,  #13
      0,  0, 70,  #14
      0, 60,100,  #15
      0, 80,100,  #16
      0,  0,230,  #17
    119, 11, 32,  #18
]
# Extend to 256*3 entries
CITYSCAPES_PALETTE += [0] * (256*3 - len(CITYSCAPES_PALETTE))

def apply_palette(mask_np: np.ndarray) -> Image.Image:
    """Convert a HxW np.uint8 mask to a paletted PIL image."""
    pil = Image.fromarray(mask_np.astype(np.uint8), mode='P')
    pil.putpalette(CITYSCAPES_PALETTE)
    return pil

class RadarSegDataset(Dataset):
    """
    Expects directory structure:
      data_dir/
        angle_range_numpy/*.npy      (float32 range–angle maps)
        masks/*.png                  (uint8 masks, same stem as npy)
        images/*.png|*.jpg (optional for overlays)

    Args:
        data_dir: root path
        image_size: (H, W) to resize both radar maps and masks
        percent_clip: tuple (low, high) percentiles for per-sample clipping before min-max
        file_stems: Optional list of stems to subset (e.g., a val split)
    """
    def __init__(self,
                 data_dir: str,
                 radar_subdir: str = 'angle_range_numpy',
                 image_size: Tuple[int, int] = (512, 512),
                 percent_clip: Tuple[float, float] = (1.0, 99.0),
                 file_stems: Optional[List[str]] = None):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.percent_clip = percent_clip

        npy_dir = os.path.join(data_dir, radar_subdir)
        mask_dir = os.path.join(data_dir, 'masks')
        self.img_dir = os.path.join(data_dir, 'images')  # optional

        npy_paths = sorted(glob.glob(os.path.join(npy_dir, '*.npy')))
        if file_stems is not None:
            stem_set = set(file_stems)
            npy_paths = [p for p in npy_paths if os.path.splitext(os.path.basename(p))[0] in stem_set]

        self.samples = []
        for npy_path in npy_paths:
            stem = os.path.splitext(os.path.basename(npy_path))[0]
            mask_path = os.path.join(mask_dir, f'{stem}.png')
            if os.path.exists(mask_path):
                self.samples.append((stem, npy_path, mask_path))

        if len(self.samples) == 0:
            raise RuntimeError(f'No matched (npy, mask) pairs found under {data_dir}')

    def __len__(self):
        return len(self.samples)

    def _normalize_radar(self, x: np.ndarray) -> np.ndarray:
        """Per-sample robust min-max to [0,1] with percentile clipping."""
        lowp, highp = np.percentile(x, self.percent_clip)
        x = np.clip(x, lowp, highp)
        if highp - lowp < 1e-6:
            return np.zeros_like(x, dtype=np.float32)
        x = (x - lowp) / (highp - lowp)
        return x.astype(np.float32)

    def _resize_pair(self, radar: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        H, W = self.image_size
        # radar: (H0, W0) float32 [0,1] → PIL → resize bilinear → back to np
        radar_img = Image.fromarray((radar * 255.0).astype(np.uint8))  # scale to 0..255 for PIL
        radar_img = radar_img.resize((W, H), resample=Image.BILINEAR)
        radar_np = np.array(radar_img).astype(np.float32) / 255.0

        mask_img = Image.fromarray(mask.astype(np.uint8))
        mask_img = mask_img.resize((W, H), resample=Image.NEAREST)
        mask_np = np.array(mask_img).astype(np.uint8)
        return radar_np, mask_np

    def __getitem__(self, idx: int):
        stem, npy_path, mask_path = self.samples[idx]
        radar = np.load(npy_path).astype(np.float32)  # (H, W) float32
        mask = np.array(Image.open(mask_path).convert('L')).astype(np.uint8)  # (H, W)

        radar = self._normalize_radar(radar)
        radar, mask = self._resize_pair(radar, mask)

        # To tensor: radar -> (1, H, W), mask -> (H, W)
        radar_t = torch.from_numpy(radar)[None, ...]  # single channel
        mask_t = torch.from_numpy(mask)

        # Optional RGB path (for overlay)
        rgb_path = None
        jpg = os.path.join(self.img_dir, f'{stem}.jpg')
        png = os.path.join(self.img_dir, f'{stem}.png')
        if os.path.exists(jpg):
            rgb_path = jpg
        elif os.path.exists(png):
            rgb_path = png

        return {
            'stem': stem,
            'radar': radar_t,  # [1,H,W]
            'mask': mask_t,    # [H,W]
            'rgb_path': rgb_path,
        }

# ----------------------
# Metrics
# ----------------------
def fast_hist(pred: np.ndarray, target: np.ndarray, num_classes: int) -> np.ndarray:
    k = (target >= 0) & (target < num_classes)
    return np.bincount(
        num_classes * target[k].astype(int) + pred[k].astype(int),
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)

def compute_miou_from_hist(hist: np.ndarray) -> float:
    # IoU = TP / (TP + FP + FN)
    with np.errstate(divide='ignore', invalid='ignore'):
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    # Ignore classes with no presence in GT (denominator 0)
    valid = ~np.isnan(iou)
    if valid.sum() == 0:
        return 0.0
    return float(np.mean(iou[valid]))

def pixel_accuracy(hist: np.ndarray) -> float:
    correct = np.diag(hist).sum()
    total = hist.sum()
    return float(correct / (total + 1e-10))
