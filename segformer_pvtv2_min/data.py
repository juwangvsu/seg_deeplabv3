import os
from PIL import Image
from typing import Tuple
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class SegDataset(Dataset):
    def __init__(self, root: str, split: str, img_dirname: str, mask_dirname: str, input_size: int, ignore_index: int = 255):
        super().__init__()
        self.root = root
        self.split = split
        self.img_dir = os.path.join(root, split, img_dirname)
        self.mask_dir = os.path.join(root, split, mask_dirname)
        self.input_size = input_size
        self.ignore_index = ignore_index
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        assert len(self.images) > 0, f"No images found in {self.img_dir}"

    def __len__(self):
        return len(self.images)

    def _random_crop_params(self, w, h, th, tw):
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def _train_transform(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        short_side = random.randint(int(0.8 * self.input_size), int(1.2 * self.input_size))
        w, h = image.size
        scale = short_side / min(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        image = image.resize((nw, nh), Image.BILINEAR)
        mask = mask.resize((nw, nh), Image.NEAREST)
        i, j, th, tw = self._random_crop_params(nw, nh, self.input_size, self.input_size)
        image = TF.crop(image, i, j, th, tw)
        mask = TF.crop(mask, i, j, th, tw)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask

    def _val_transform(self, image: Image.Image, mask: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        w, h = image.size
        scale = self.input_size / min(w, h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        image = image.resize((nw, nh), Image.BILINEAR)
        mask = mask.resize((nw, nh), Image.NEAREST)
        i = max(0, (nh - self.input_size) // 2)
        j = max(0, (nw - self.input_size) // 2)
        image = TF.crop(image, i, j, self.input_size, self.input_size)
        mask = TF.crop(mask, i, j, self.input_size, self.input_size)
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask

    def __getitem__(self, idx: int):
        name = self.images[idx]
        img_path = os.path.join(self.img_dir, name)
        base, _ = os.path.splitext(name)
        mask_path = None
        for ext in ['.png', '.jpg']:
            candidate = os.path.join(self.mask_dir, base + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        if mask_path is None:
            raise FileNotFoundError(f"Mask not found for image {name} in {self.mask_dir}")

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        if self.split == 'train':
            image, mask = self._train_transform(image, mask)
        else:
            image, mask = self._val_transform(image, mask)

        return {'image': image, 'mask': mask, 'name': base}
