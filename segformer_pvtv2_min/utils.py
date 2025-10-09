import os
from typing import Dict
import yaml
import torch
from PIL import Image
import numpy as np

CITYSCAPES_19 = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

def load_config(path: str) -> Dict:
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg

def save_checkpoint(state: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def colorize_mask(mask: np.ndarray, palette=CITYSCAPES_19) -> Image.Image:
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, rgb in enumerate(palette):
        color[mask == cid] = rgb
    return Image.fromarray(color)

def overlay_image(image: Image.Image, color_mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    image = image.convert('RGBA')
    color_mask = color_mask.convert('RGBA')
    return Image.blend(image, color_mask, alpha)
