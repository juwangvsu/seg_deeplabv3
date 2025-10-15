#!/usr/bin/env python3
# infer_segformer.py

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from transformers import (
    AutoImageProcessor,
    SegformerForSemanticSegmentation,
)

# ----------------------------
# Palettes
# ----------------------------

def palette_ade20k() -> List[Tuple[int,int,int]]:
    # 150 ADE20K colors (trimmed if model has fewer classes)
    # Source: common ADE20K palette
    ade = [
        (120,120,120),(180,120,120),(6,230,230),(80,50,50),(4,200,3),
        (120,120,80),(140,140,140),(204,5,255),(230,230,230),(4,250,7),
        (224,5,255),(235,255,7),(150,5,61),(120,120,70),(8,255,51),
        (255,6,82),(143,255,140),(204,255,4),(255,51,7),(204,70,3),
        (0,102,200),(61,230,250),(255,6,51),(11,102,255),(255,7,71),
        (255,9,224),(9,7,230),(220,220,220),(255,9,92),(112,9,255),
        (8,255,214),(7,255,224),(255,184,6),(10,255,71),(255,41,10),
        (7,255,255),(224,255,8),(102,8,255),(255,61,6),(255,194,7),
        (255,122,8),(0,255,20),(255,8,41),(255,5,153),(6,51,255),
        (235,12,255),(160,150,20),(0,163,255),(140,140,140),(250,10,15),
        (20,255,0),(31,255,0),(255,31,0),(255,224,0),(153,255,0),
        (0,0,255),(255,71,0),(0,235,255),(0,173,255),(31,0,255),
        (11,200,200),(255,82,0),(0,255,245),(0,61,255),(0,255,112),
        (0,255,133),(255,0,0),(255,163,0),(255,102,0),(194,255,0),
        (0,143,255),(51,255,0),(0,82,255),(0,255,41),(0,255,173),
        (10,0,255),(173,255,0),(0,255,153),(255,92,0),(255,0,255),
        (255,0,245),(255,0,102),(255,173,0),(255,0,20),(255,184,184),
        (0,31,255),(0,255,61),(0,71,255),(255,0,204),(0,255,194),
        (0,255,82),(0,10,255),(0,112,255),(51,0,255),(0,194,255),
        (0,122,255),(0,255,163),(255,153,0),(0,255,10),(255,112,0),
        (143,255,0),(82,0,255),(163,255,0),(255,235,0),(8,184,170),
        (133,0,255),(0,255,92),(184,0,255),(255,0,31),(0,184,255),
        (0,214,255),(255,0,112),(92,255,0),(0,224,255),(112,224,255),
        (70,184,160),(163,0,255),(153,0,255),(71,255,0),(255,0,163),
        (255,204,0),(255,0,143),(0,255,235),(133,255,0),(255,0,235),
        (245,0,255),(255,0,122),(255,245,0),(10,190,212),(214,255,0),
        (0,204,255),(20,0,255),(255,255,0),(0,153,255),(0,41,255),
        (0,255,204),(41,0,255),(41,255,0),(173,0,255),(0,245,255),
        (71,0,255),(122,0,255),(0,255,184),(0,92,255),(184,255,0),
        (0,133,255),(255,214,0),(25,194,194),(102,255,0),(92,0,255)
    ]
    return ade

def palette_cityscapes() -> List[Tuple[int,int,int]]:
    # 19 Cityscapes eval classes
    return [
        (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
        (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
        ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
        (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32),
    ]

def palette_grayscale(n: int) -> List[Tuple[int,int,int]]:
    return [(i, i, i) for i in np.linspace(0, 255, num=n, dtype=np.uint8)]

def resolve_palette(name: str, n_classes: int) -> List[Tuple[int,int,int]]:
    name = name.lower()
    if name == "ade20k":
        pal = palette_ade20k()
    elif name == "cityscapes":
        pal = palette_cityscapes()
    elif name == "grayscale":
        pal = palette_grayscale(n_classes)
    else:
        raise ValueError(f"Unknown palette: {name}")
    if len(pal) < n_classes:
        # Repeat as needed
        reps = (n_classes + len(pal) - 1) // len(pal)
        pal = (pal * reps)[:n_classes]
    else:
        pal = pal[:n_classes]
    return pal

# ----------------------------
# IO helpers
# ----------------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def list_images(path: Path) -> List[Path]:
    if path.is_dir():
        files = sorted([p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS])
    else:
        files = [path]
    if not files:
        raise FileNotFoundError(f"No images found at {path}")
    return files

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def colorize_mask(mask: np.ndarray, palette: List[Tuple[int,int,int]]) -> Image.Image:
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, (r, g, b) in enumerate(palette):
        rgb[mask == idx] = (r, g, b)
    return Image.fromarray(rgb, mode="RGB")

def overlay_image(img: Image.Image, color_mask: Image.Image, alpha: float=0.5) -> Image.Image:
    img_rgb = img.convert("RGB")
    color_mask = color_mask.resize(img_rgb.size, Image.NEAREST)
    return Image.blend(img_rgb, color_mask, alpha)

# ----------------------------
# Inference
# ----------------------------

@torch.no_grad()
def run_batch(
    model: SegformerForSemanticSegmentation,
    processor: AutoImageProcessor,
    images: List[Image.Image],
    device: torch.device,
    keep_size: bool,
):
    # Prepare inputs; processor handles resize/normalize by model defaults
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits  # [B, C, H', W']

    # Up-sample to original size(s)
    if keep_size:
        # If mixed sizes, upsample individually
        preds = []
        for i, img in enumerate(images):
            logit_i = logits[i:i+1]
            up = F.interpolate(logit_i, size=img.size[::-1], mode="bilinear", align_corners=False)
            pred = up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            preds.append(pred)
        return preds
    else:
        # Use processor postprocessing to target original sizes
        target_sizes = [img.size[::-1] for img in images]  # (H, W)
        preds = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
        # Returns list of dicts or list of tensors depending on version; normalize to np arrays
        np_preds = []
        for p in preds:
            if isinstance(p, dict) and "segmentation" in p:
                arr = p["segmentation"].cpu().numpy().astype(np.uint8)
            else:
                arr = p.cpu().numpy().astype(np.uint8)
            np_preds.append(arr)
        return np_preds

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="SegFormer inference with Hugging Face Transformers")
    ap.add_argument("--model-id", type=str, required=True,
                    help="Hub ID (e.g., nvidia/segformer-b2-finetuned-ade-512-512) or local directory.")
    ap.add_argument("--input", type=str, required=True, help="Image file or directory.")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory.")
    ap.add_argument("--palette", type=str, default="ade20k", choices=["ade20k", "cityscapes", "grayscale"],
                    help="Color palette for visualization.")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size for inference.")
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                    help="Device: cuda or cpu")
    ap.add_argument("--keep-original-size", action="store_true",
                    help="If set, resize predictions back to each image's original resolution.")
    ap.add_argument("--overlay-alpha", type=float, default=0.5, help="Alpha for overlay blend.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_root = Path(args.out_dir)
    out_masks = out_root / "masks"            # raw label maps (png, uint8)
    out_colors = out_root / "color_mask"      # colorized masks
    out_overlay = out_root / "overlay"        # blended overlays

    for d in (out_masks, out_colors, out_overlay):
        ensure_dir(d)

    device = torch.device(args.device)

    # Load model + processor
    try:
        model = SegformerForSemanticSegmentation.from_pretrained(args.model_id)
        processor = AutoImageProcessor.from_pretrained(args.model_id)
    except Exception as e:
        print(f"Failed to load model/processor from {args.model_id}: {e}", file=sys.stderr)
        sys.exit(1)

    model.to(device)
    model.eval()

    # Classes and palette
    num_labels = int(model.config.num_labels)
    palette = resolve_palette(args.palette, num_labels)

    # Label names (optional, used in filename suffix)
    id2label: Dict[int, str] = getattr(model.config, "id2label", None) or {}
    id2label = {int(k): v for k, v in id2label.items()} if isinstance(id2label, dict) else {}

    # Gather images
    images = list_images(in_path)

    # Batched inference
    bs = max(1, args.batch_size)
    for i in range(0, len(images), bs):
        chunk = images[i:i+bs]
        pil_list = [Image.open(p).convert("RGB") for p in chunk]

        preds = run_batch(
            model=model,
            processor=processor,
            images=pil_list,
            device=device,
            keep_size=args.keep_original_size,
        )

        for img_path, pil_img, pred in zip(chunk, pil_list, preds):
            # Save raw mask (uint8 PNG)
            mask_img = Image.fromarray(pred, mode="L")
            mask_save = out_masks / (img_path.stem + ".png")
            mask_img.save(mask_save)

            # Save colorized mask
            color_img = colorize_mask(pred, palette)
            color_save = out_colors / (img_path.stem + "_color.png")
            color_img.save(color_save)

            # Save overlay
            overlay = overlay_image(pil_img, color_img, alpha=args.overlay_alpha)
            overlay_save = out_overlay / (img_path.stem + "_overlay.png")
            overlay.save(overlay_save)

            # Optional: print brief stats
            unique, counts = np.unique(pred, return_counts=True)
            topk = sorted(zip(unique.tolist(), counts.tolist()), key=lambda x: x[1], reverse=True)[:5]
            topk_str = ", ".join(
                f"{id2label.get(c, str(c))}:{cnt}" for c, cnt in topk
            )
            print(f"[{img_path.name}] -> classes(top5): {topk_str}")

    print(f"Done. Outputs saved under: {out_root.resolve()}")

if __name__ == "__main__":
    main()

