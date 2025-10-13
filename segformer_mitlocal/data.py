from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class SegFolder(Dataset):
    """Folder dataset with
    root/images/*.png|jpg, root/masks/*.png (class ids [0..C-1])
    """
    def __init__(self, root: str, image_size: int = 512, aug: bool = True):
        self.root = Path(root)
        self.imgs = sorted((self.root / "images").glob("*"))
        self.masks = sorted((self.root / "masks").glob("*"))
        assert len(self.imgs) == len(self.masks) and len(self.imgs) > 0, "mismatched or empty dataset"

        if aug:
            self.transform = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self.mask_transform = T.Compose([
            T.Resize((image_size, image_size), interpolation=T.InterpolationMode.NEAREST),
            T.PILToTensor(),
        ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        img_t = self.transform(img)
        mask_t = self.mask_transform(mask).long().squeeze(0)
        return img_t, mask_t
