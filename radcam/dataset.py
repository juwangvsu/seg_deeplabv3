from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from torchvision import transforms as T
from PIL import Image
import numpy as np
import torch
import random
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
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype('float32')
        if arr.ndim == 2:
            arr = arr[None]  # [1,R,A]
        tensor = torch.from_numpy(arr).float().unsqueeze(0)  # [1,1,R,A]
        tensor = torch.nn.functional.interpolate(
            tensor, size=self.radar_size, mode="bilinear", align_corners=False
        )[0]  # [1,R',A']
        return tensor

    def __getitem__(self, idx):
        img_path, radar_path, mask_path = self.samples[idx]
        stem = img_path.stem

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

        return {"image": img, "radar": radar, "mask": mask, "stem": stem}

