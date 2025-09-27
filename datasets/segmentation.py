import os
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset

class SegDataset(Dataset):
    def __init__(self, img_dir, mask_dir, input_size=None, augment=None, eval_resize_long=None, ignore_index=255):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*")))
        self.mask_paths = [os.path.join(mask_dir, os.path.basename(p)) for p in self.img_paths]
        self.input_size = input_size
        self.augment = augment
        self.eval_resize_long = eval_resize_long
        self.ignore_index = ignore_index

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # eval-time: resize by long side while keeping aspect
        if self.eval_resize_long is not None and self.input_size is None:
            h, w = img.shape[:2]
            scale = self.eval_resize_long / max(h, w)
            if scale != 1.0:
                nh, nw = int(round(h * scale)), int(round(w * scale))
                img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
                mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)

        if self.augment is not None:
            augmented = self.augment(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]

        if self.input_size is not None:
            # center or random crop handled by augment; ensure final size
            ih, iw = img.shape[:2]
            th, tw = self.input_size, self.input_size
            if ih < th or iw < tw:
                pad_h = max(0, th - ih)
                pad_w = max(0, tw - iw)
                img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
                mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=self.ignore_index)
            # random crop
            ih, iw = img.shape[:2]
            y = np.random.randint(0, ih - th + 1)
            x = np.random.randint(0, iw - tw + 1)
            img = img[y:y+th, x:x+tw]
            mask = mask[y:y+th, x:x+tw]

        # to tensor (HWC->CHW)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (0,1,2))  # keep HWC for Albumentations normalization if used later
        # normalize with ImageNet stats
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        img = np.transpose(img, (2,0,1))  # CHW

        return {
            "image": img,
            "mask": mask.astype(np.int64),
            "path": os.path.basename(img_path),
        }
