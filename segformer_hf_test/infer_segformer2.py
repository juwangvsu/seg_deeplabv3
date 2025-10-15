#!/usr/bin/env python3
# infer_segformer.py — Cityscapes eval (accuracy + mIoU) with saving colored masks & overlays

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# ----------------------------
# Cityscapes constants
# ----------------------------

CITYSCAPES_CLASSES = 19
CITYSCAPES_IGNORE_INDEX = 255

def cityscapes_palette() -> List[Tuple[int, int, int]]:
    # 19 Cityscapes eval classes in trainId order
    return [
        (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
        (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
        ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
        (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32),
    ]

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# ----------------------------
# Helpers
# ----------------------------

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
    if color_mask.size != img_rgb.size:
        color_mask = color_mask.resize(img_rgb.size, Image.NEAREST)
    return Image.blend(img_rgb, color_mask, alpha)

def list_cityscapes_images(input_dir: Path) -> List[Path]:
    # Prefer the Cityscapes naming *_leftImg8bit.png; if none found, fall back to all images.
    imgs = sorted(input_dir.rglob("*_leftImg8bit.png"))
    if not imgs:
        imgs = sorted([p for p in input_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])
    if not imgs:
        raise FileNotFoundError(f"No images found under {input_dir}")
    return imgs

def match_cityscapes_mask(img_path: Path, input_dir: Path, mask_dir: Path) -> Optional[Path]:
    """
    Given .../<split>/<city>/<stem>_leftImg8bit.png, try:
      .../gtFine/<split>/<city>/<stem>_gtFine_labelTrainIds.png
      .../gtFine/<split>/<city>/<stem>_gtFine_labelIds.png
    Works even if input_dir/mask_dir aren't named exactly 'leftImg8bit'/'gtFine';
    it preserves the relative split/city/filename relationship where possible.
    """
    rel = img_path.relative_to(input_dir)

    # strip suffix "_leftImg8bit" from filename stem if present
    stem = img_path.stem
    base = stem.replace("_leftImg8bit", "")

    # try to preserve split/city if present (expect rel like: <split>/<city>/<file>)
    if len(rel.parts) >= 3:
        split, city = rel.parts[-3], rel.parts[-2]
        cand1 = mask_dir / split / city / f"{base}_gtFine_labelTrainIds.png"
        cand2 = mask_dir / split / city / f"{base}_gtFine_labelIds.png"
        if cand1.exists(): return cand1
        if cand2.exists(): return cand2

    # fallback: search under mask_dir for matching basename
    cand1_list = list(mask_dir.rglob(f"{base}_gtFine_labelTrainIds.png"))
    if cand1_list: return cand1_list[0]
    cand2_list = list(mask_dir.rglob(f"{base}_gtFine_labelIds.png"))
    if cand2_list: return cand2_list[0]
    return None

def fast_hist(pred: np.ndarray, tgt: np.ndarray, n_class: int, ignore_index: int) -> np.ndarray:
    mask = (tgt != ignore_index) & (tgt>=0) & (tgt < n_class)
    if mask.sum() == 0:
        return np.zeros((n_class, n_class), dtype=np.int64)
    hist = np.bincount(
        (n_class * tgt[mask].astype(np.int64) + pred[mask]).astype(np.int64),
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist

def compute_metrics(hist: np.ndarray) -> Tuple[float, float, np.ndarray]:
    # Overall pixel accuracy (ignoring 255 handled before hist aggregation)
    acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + 1e-10)
    valid = ~np.isnan(iu)
    miou = iu[valid].mean() if valid.any() else float("nan")
    return float(acc), float(miou), iu

# ----------------------------
# Inference (batched)
# ----------------------------

@torch.no_grad()
def run_batch(
    model: SegformerForSemanticSegmentation,
    processor: AutoImageProcessor,
    images: List[Image.Image],
    target_sizes: List[Tuple[int, int]],  # (H, W) per image
    device: torch.device,
) -> List[np.ndarray]:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # Post-process directly to GT mask sizes
    post = processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)
    preds = []
    for p in post:
        if isinstance(p, dict) and "segmentation" in p:
            arr = p["segmentation"].cpu().numpy().astype(np.uint8)
        elif torch.is_tensor(p):
            arr = p.cpu().numpy().astype(np.uint8)
        else:
            # unexpected; upsample manually to target size
            logit = outputs.logits[preds.__len__():preds.__len__()+1]
            H, W = target_sizes[preds.__len__()]
            up = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
            arr = up.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        preds.append(arr)
    return preds

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="SegFormer inference + Cityscapes accuracy/mIoU")
    ap.add_argument("--model-id", required=True, type=str,
                    help="Hugging Face model id or local path (e.g., nvidia/segformer-b2-finetuned-cityscapes-1024-1024)")
    ap.add_argument("--input-dir", required=True, type=str,
                    help="Directory containing Cityscapes images (leftImg8bit format).")
    ap.add_argument("--mask-dir", required=True, type=str,
                    help="Directory containing Cityscapes GT masks (gtFine format).")
    ap.add_argument("--out-dir", required=True, type=str, help="Output root directory.")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overlay-alpha", type=float, default=0.5)
    ap.add_argument('--showmodel', action='store_true')

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    out_root = Path(args.out_dir)
    out_masks = out_root / "masks"
    out_colors = out_root / "color_mask"
    out_overlay = out_root / "overlay"
    for d in (out_masks, out_colors, out_overlay):
        ensure_dir(d)

    device = torch.device(args.device)
    model = SegformerForSemanticSegmentation.from_pretrained(args.model_id).to(device).eval()
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    print('xxx ', model)
    if args.showmodel:
        exit(0)
    num_labels = int(model.config.num_labels)
    if num_labels != CITYSCAPES_CLASSES:
        print(f"[warn] Model num_labels={num_labels}, but Cityscapes eval uses {CITYSCAPES_CLASSES}. Proceeding anyway.")
    palette = cityscapes_palette()

    images = list_cityscapes_images(input_dir)

    # Metrics accumulators
    hist = np.zeros((CITYSCAPES_CLASSES, CITYSCAPES_CLASSES), dtype=np.int64)

    # Iterate in batches
    B = max(1, args.batch_size)
    i = 0
    total = len(images)
    processed = 0

    while i < total:
        batch_paths = images[i:i+B]
        i += len(batch_paths)

        # Match masks; drop any images without a matching GT
        paired = []
        for p in batch_paths:
            m = match_cityscapes_mask(p, input_dir, mask_dir)
            if m is None:
                print(f"[skip] No GT mask for: {p}")
            else:
                paired.append((p, m))

        if not paired:
            continue

        # Load images and masks
        pil_imgs: List[Image.Image] = [Image.open(p).convert("RGB") for p, _ in paired]
        gt_masks: List[np.ndarray] = []
        target_sizes: List[Tuple[int,int]] = []
        for _, mpath in paired:
            gt = np.array(Image.open(mpath), dtype=np.uint8)
            # If labelIds provided, you may need mapping -> trainIds; here we trust dataset uses labelTrainIds already.
            gt_masks.append(gt)
            target_sizes.append(gt.shape)  # (H, W)

        # Predict at GT sizes
        preds = run_batch(model, processor, pil_imgs, target_sizes, device)

        # Save outputs + update metrics
        for (img_path, mask_path), img, pred, gt in zip(paired, pil_imgs, preds, gt_masks):
            # Ensure pred shape matches GT
            if pred.shape != gt.shape:
                # resize prediction to gt
                pred = np.array(Image.fromarray(pred, mode="L").resize(gt.shape[::-1], Image.NEAREST), dtype=np.uint8)

            # Metrics (ignore 255)
            hist += fast_hist(pred, gt, CITYSCAPES_CLASSES, CITYSCAPES_IGNORE_INDEX)

            # Paths preserving relative structure under input_dir
            rel = img_path.relative_to(input_dir)
            # build out paths (replace extension with .png)
            pred_raw_path = (out_masks / rel).with_suffix(".png")
            pred_col_path = (out_colors / rel).with_suffix(".png")
            pred_ovl_path = (out_overlay / rel).with_suffix(".png")
            for p in (pred_raw_path.parent, pred_col_path.parent, pred_ovl_path.parent):
                ensure_dir(p)

            # Save raw
            Image.fromarray(pred, mode="L").save(pred_raw_path)

            # Save colored + overlay
            color = colorize_mask(pred, palette)
            color.save(pred_col_path)
            ovl = overlay_image(img, color, alpha=args.overlay_alpha)
            ovl.save(pred_ovl_path)

            processed += 1
            print(f"[{processed}/{total}] {img_path.name} -> saved mask/color/overlay")

    # Final metrics
    acc, miou, per_class_iou = compute_metrics(hist)
    print("\n========== Cityscapes Evaluation ==========")
    print(f"Pixel Accuracy: {acc*100:.2f}%")
    print(f"Mean IoU:       {miou*100:.2f}%")
    # Uncomment if you want per-class IoU:
    # class_names = ["road","sidewalk","building","wall","fence","pole","traffic light","traffic sign",
    #                "vegetation","terrain","sky","person","rider","car","truck","bus","train","motorcycle","bicycle"]
    # for idx, iou in enumerate(per_class_iou):
    #     print(f"  {idx:2d} {class_names[idx]:>14}: {iou*100:.2f}%")

    print("===========================================")
    print(f"Outputs saved under: {out_root.resolve()}")

if __name__ == "__main__":
    main()

