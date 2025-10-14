
from pathlib import Path
from typing import Literal, Tuple, List
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

DatasetType = Literal["folder", "cityscapes"]

class SegFolder(Dataset):
    """Generic segmentation dataset with two modes:

    1) dataset="folder":
         root/
           images/*.jpg|png
           masks/*.png    # class ids [0..C-1]

    2) dataset="cityscapes":
         root/
           leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
           gtFine/{train,val,test}/<city>/*_gtFine_labelTrainIds.png  (preferred)
                                                    or *_gtFine_labelIds.png
         - If only *_labelIds.png is present, we map labelIds → trainIds (0..18, 255=ignore)
    """

    def __init__(
        self,
        root: str,
        image_size: Tuple[int,int]=(512,896), #int = 512,
        aug: bool = True,
        dataset: DatasetType = "folder",
        split: Literal["train", "val", "test"] = "train",
    ):
        self.root = Path(root)
        self.dataset = dataset
        self.split = split
        self.image_size = image_size

        if aug:
            self.transform = T.Compose([
                T.Resize((image_size[0], image_size[1]), interpolation=T.InterpolationMode.BILINEAR),
                T.RandomHorizontalFlip(),
                T.ColorJitter(0.2, 0.2, 0.2, 0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size[0], image_size[1]), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        self._mask_resize = T.Resize((image_size[0], image_size[1]), interpolation=T.InterpolationMode.NEAREST)

        if dataset == "folder":
            self.imgs = sorted((self.root / "images").glob("*"))
            self.masks = sorted((self.root / "masks").glob("*"))
            assert len(self.imgs) == len(self.masks) and len(self.imgs) > 0, "mismatched or empty dataset"
            self._needs_cityscapes_map = False
        elif dataset == "cityscapes":
            self.imgs, self.masks, self._needs_cityscapes_map = self._scan_cityscapes()
            assert len(self.imgs) > 0, f"No Cityscapes data found in {self.root} for split={split}"
        else:
            raise ValueError(f"Unknown dataset type: {dataset}")

    def _scan_cityscapes(self) -> Tuple[List[Path], List[Path], bool]:
        imgs_root = self.root / "leftImg8bit" / self.split
        gtf_root = self.root / "gtFine" / self.split
        img_files: List[Path] = []
        mask_files: List[Path] = []
        needs_map_any = False
        for city_dir in sorted(imgs_root.glob("*")):
            if not city_dir.is_dir():
                continue
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                stem = img_path.name.replace("_leftImg8bit.png", "")
                mask_dir = gtf_root / city_dir.name
                m_train = mask_dir / f"{stem}_gtFine_labelTrainIds.png"
                m_label = mask_dir / f"{stem}_gtFine_labelIds.png"
                if m_train.exists():
                    img_files.append(img_path)
                    mask_files.append(m_train)
                elif m_label.exists():
                    img_files.append(img_path)
                    mask_files.append(m_label)
                    needs_map_any = True
        return img_files, mask_files, needs_map_any

    @staticmethod
    def _labelids_to_trainids(arr: np.ndarray) -> np.ndarray:
        lut = np.full(256, 255, dtype=np.uint8)
        mapping = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5,
            19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
            24: 11, 25: 12, 26: 13, 27: 14, 28: 15,
            31: 16, 32: 17, 33: 18,
        }
        for k, v in mapping.items():
            lut[k] = v
        return lut[arr]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img = Image.open(self.imgs[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])

        img_t = self.transform(img)

        mask = self._mask_resize(mask)
        mask_np = np.array(mask, dtype=np.uint8)
        if self.dataset == "cityscapes" and self._needs_cityscapes_map:
            mask_np = self._labelids_to_trainids(mask_np)
        mask_t = torch.from_numpy(mask_np.astype(np.int64))

        return img_t, mask_t
