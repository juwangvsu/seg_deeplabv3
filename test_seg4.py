#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List
import colorsys

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


def build_palette(n: int, seed: int = 0) -> np.ndarray:
    """Deterministic [n,3] RGB palette (0..255)."""
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    sat = 0.75
    val = 0.95
    colors = []
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(float(h), sat, val)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    colors = np.array(colors, dtype=np.uint8)
    return colors[rng.permutation(n)]


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map class IDs in HxW mask to HxWx3 RGB using palette."""
    mask = mask.astype(np.int64)
    mask = np.clip(mask, 0, palette.shape[0] - 1)
    return palette[mask]


def load_labels(model) -> List[str]:
    """Get contiguous id->label list from HF model config."""
    id2label_raw: Dict[str, str] = getattr(model.config, "id2label", {})
    if id2label_raw:
        id2label = {int(k): v for k, v in id2label_raw.items()}
        num_labels = max(id2label.keys()) + 1
        labels = [id2label.get(i, f"class_{i}") for i in range(num_labels)]
    else:
        num_labels = getattr(model.config, "num_labels", 150)
        labels = [f"class_{i}" for i in range(num_labels)]
    return labels


def segment_one(
    img_path: Path,
    overlay_dir: Path,
    colored_dir: Path,
    masks_dir: Path,
    processor,
    model,
    labels: List[str],
    alpha: float,
    max_legend: int,
    device: torch.device,
    no_legend: bool,
) -> None:
    img = Image.open(img_path).convert("RGB")
    W, H = img.size

    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # [1, C, h, w]
    up = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    pred = up.argmax(1)[0].cpu().numpy().astype(np.uint16)  # HxW with class IDs

    # Prepare outputs
    stem = img_path.stem
    overlay_out = overlay_dir / f"{stem}_overlay.png"
    colored_out = colored_dir / f"{stem}_color.png"
    mask_out = masks_dir / f"{stem}_mask.png"

    # Palette + colorized mask
    palette = build_palette(len(labels), seed=0) #city scrape 19 classes
    print('xxx palette ', palette)
    color_mask = colorize_mask(pred, palette)
    color_img = Image.fromarray(color_mask, mode="RGB")
    color_img.save(colored_out)

    # Raw mask (8-bit is fine for <=255 classes; ADE20K=150)
    Image.fromarray(pred.astype(np.uint8), mode="L").save(mask_out)

    # Blend overlay
    blend = Image.blend(img, color_img, alpha=float(alpha))

    # Legend: show only classes present, sorted by frequency
    uniq = np.unique(pred).tolist()
    print('xxx uniq ', uniq)
    uniq = [i for i in uniq if 0 <= i < len(labels)]
    counts = [(i, int((pred == i).sum())) for i in uniq]
    counts.sort(key=lambda x: x[1], reverse=True)
    show_ids = [i for i, _ in counts[:max_legend]]

    # Plot and save overlay PNG
    plt.figure(figsize=(16, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.set_title("Original")
    ax1.axis("off")

    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(blend)
    ax2.set_title("Segmentation (overlay)")
    ax2.axis("off")

    if not no_legend and len(show_ids) > 0:
        legend_patches = [
            Patch(facecolor=palette[i] / 255.0, edgecolor="black", label=f"{i}: {labels[i]}")
            for i in show_ids
        ]
        ax2.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            title="Classes",
        )

    plt.tight_layout()
    plt.savefig(overlay_out, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"✔ {img_path.name} ->")
    print(f"    overlay:      {overlay_out}")
    print(f"    colored mask: {colored_out}")
    print(f"    raw mask:     {mask_out}")


def main():
    p = argparse.ArgumentParser(
        description="Batch semantic segmentation (HF Transformers) with labeled overlays and raw masks."
    )
    p.add_argument("--data_dir", required=True, help="Data root dir containing an 'images' subfolder.")
    p.add_argument("--out_dir", default="outputs/segformer", help="Data dir output subfolder.")
    p.add_argument("--input", default="", help="Optional single image name or path. If omitted, process all images in <data_dir>/images.")
    #p.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512",
    p.add_argument("--model", default="nvidia/segformer-b5-finetuned-cityscapes-1024-1024",
                   help="HF model repo id for semantic segmentation.")
    p.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha ∈ [0,1].")
    p.add_argument("--max-legend", type=int, default=25, help="Max classes listed in legend.")
    p.add_argument("--device", default="", help='Force "cuda" or "cpu". Default: auto.')
    p.add_argument("--no-legend", action="store_true", help="Disable legend.")
    p.add_argument("--plot", action="store_true", help="Also display plots interactively (single-image mode).")
    args = p.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    out_root = Path(args.out_dir).expanduser().resolve()
    images_dir = root / "images"
    overlay_dir = out_root / "overlay"
    colored_dir = out_root / "colored_masks"
    masks_dir = out_root / "masks"
    for d in (overlay_dir, colored_dir, masks_dir):
        d.mkdir(parents=True, exist_ok=True)

    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    print('xxx device ', dev)
    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model).to(dev).eval()
    labels = load_labels(model)

    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")

    targets: List[Path] = []
    if args.input:
        in_path = Path(args.input)
        if not in_path.is_file():
            # Treat as a name inside images_dir
            in_path = images_dir / args.input
        if not in_path.is_file():
            raise FileNotFoundError(f"Input not found: {args.input} (looked at {in_path})")
        targets = [in_path]
    else:
        for pat in exts:
            targets.extend(images_dir.glob(pat))
        targets = sorted(targets)
        if not targets:
            raise FileNotFoundError(f"No images found under {images_dir} with extensions: {', '.join(exts)}")

    for i, img_path in enumerate(targets, 1):
        segment_one(
            img_path=img_path,
            overlay_dir=overlay_dir,
            colored_dir=colored_dir,
            masks_dir=masks_dir,
            processor=processor,
            model=model,
            labels=labels,
            alpha=args.alpha,
            max_legend=args.max_legend,
            device=dev,
            no_legend=args.no_legend,
        )
        if args.plot and len(targets) == 1:
            # If interactive plot requested and only one image, show it.
            import matplotlib.pyplot as plt  # re-import safe
            plt.show()


if __name__ == "__main__":
    main()

