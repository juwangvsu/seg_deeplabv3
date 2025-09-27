import os
import time
import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from datasets.segmentation import SegDataset
from models.deeplab import create_deeplabv3_resnet50
from utils.metrics import compute_mIoU, pixel_accuracy
import albumentations as A

def get_augment(cfg):
    aug = cfg.get("augment", {})
    min_s, max_s = aug.get("min_scale", 0.5), aug.get("max_scale", 1.5)
    hflip = aug.get("hflip_prob", 0.5)
    cj = aug.get("color_jitter", [0.2,0.2,0.2,0.1])
    transforms = A.Compose([
        A.LongestMaxSize(max_size=int(1028 * max_s), interpolation=1),
        A.RandomScale(scale_limit=(min_s-1.0, max_s-1.0), p=1.0),
        A.HorizontalFlip(p=hflip),
        A.ColorJitter(brightness=cj[0], contrast=cj[1], saturation=cj[2], hue=cj[3], p=0.8),
    ])
    return transforms

def make_dataloaders(cfg, input_size=None, eval_resize_long=None, shuffle=True):
    dcfg = cfg["data"]
    root = dcfg["root"]
    train_ds = SegDataset(
        img_dir=os.path.join(root, dcfg["train_images"]),
        mask_dir=os.path.join(root, dcfg["train_masks"]),
        input_size=input_size, augment=get_augment(cfg), eval_resize_long=None, ignore_index=cfg["train"]["ignore_index"]
    )
    val_ds = SegDataset(
        img_dir=os.path.join(root, dcfg["val_images"]),
        mask_dir=os.path.join(root, dcfg["val_masks"]),
        input_size=None, augment=None, eval_resize_long=eval_resize_long, ignore_index=cfg["train"]["ignore_index"]
    )
    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=shuffle, num_workers=cfg["train"]["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)
    return train_loader, val_loader

def save_checkpoint(state, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(state, out_path)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
