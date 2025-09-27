#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation


def build_palette(n: int, seed: int = 0) -> np.ndarray:
    """
    Create a deterministic color palette of shape [n, 3] with values in 0..255.
    """
    rng = np.random.default_rng(seed)
    # Start with distinct hues, then shuffle slightly for variety
    hues = np.linspace(0, 1, n, endpoint=False)
    sat = 0.75
    val = 0.95
    import colorsys
    colors = []
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(float(h), sat, val)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    colors = np.array(colors, dtype=np.uint8)
    # Small deterministic permutation for variety
    perm = rng.permutation(n)
    return colors[perm]


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """
    Map class IDs in mask to RGB colors using palette.
    mask: HxW (uint16/uint8)
    palette: [num_classes, 3]
    """
    mask = mask.astype(np.int64)
    mask = np.clip(mask, 0, palette.shape[0] - 1)
    return palette[mask]


def run_segmentation(
    input_path: str,
    output_plot: str,
    output_mask: str,
    output_colored: str,
    model_name: str,
    alpha: float,
    max_legend: int,
    device: str,
    no_legend: bool,
) -> None:
    # Device
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model + processor
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name).to(dev).eval()

    # Labels
    # id2label is {int_id: "class_name"}; ensure contiguous mapping
    id2label_raw: Dict[str, str] = getattr(model.config, "id2label", {})
    # keys might be str; coerce to int and rebuild contiguous list
    if id2label_raw:
        id2label = {int(k): v for k, v in id2label_raw.items()}
        num_labels = max(id2label.keys()) + 1
        labels = [id2label.get(i, f"class_{i}") for i in range(num_labels)]
    else:
        # Fallback
        num_labels = getattr(model.config, "num_labels", 150)
        labels = [f"class_{i}" for i in range(num_labels)]

    palette = build_palette(len(labels), seed=0)

    # Load image
    img = Image.open(input_path).convert("RGB")
    W, H = img.size

    # Preprocess + forward
    inputs = processor(images=img, return_tensors="pt").to(dev)
    with torch.no_grad():
        outputs = model(**inputs)

    # Upsample logits to input size
    logits = outputs.logits  # [1, C, h, w]
    up = torch.nn.functional.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    pred = up.argmax(1)[0].cpu().numpy().astype(np.uint16)  # HxW

    # ---- Save raw mask (class IDs) ----
    # PNG supports up to 16-bit; ADE20K has 150 classes => uint8 is fine; keep uint16 for generality.
    mask_arr = pred
    mask_img = Image.fromarray(mask_arr.astype(np.uint16), mode="I;16")
    # If you prefer 8-bit indexed: uncomment the next two lines and save as "P" mode.
    # mask_img = Image.fromarray(mask_arr.astype(np.uint8), mode="L")
    mask_img.save(output_mask)

    # ---- Create colorized overlay ----
    color_mask = colorize_mask(mask_arr, palette)  # HxWx3, uint8
    color_img = Image.fromarray(color_mask, mode="RGB")
    if output_colored:
        color_img.save(output_colored)

    # Blend with original
    blend = Image.blend(img, color_img, alpha=float(alpha))

    # ---- Build legend (only detected classes) ----
    uniq_ids: List[int] = np.unique(mask_arr).tolist()
    # Filter out-of-range just in case
    uniq_ids = [i for i in uniq_ids if 0 <= i < len(labels)]
    # Sort by frequency (descending) for a useful legend
    counts = [(i, int((mask_arr == i).sum())) for i in uniq_ids]
    counts.sort(key=lambda x: x[1], reverse=True)
    show_ids = [i for i, _ in counts[:max_legend]]

    # ---- Plot & save ----
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
        # Put the legend outside the image
        ax2.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            title="Classes",
        )

    plt.tight_layout()
    plt.savefig(output_plot, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"✔ Saved plot: {output_plot}")
    print(f"✔ Saved raw mask (IDs): {output_mask}")
    if output_colored:
        print(f"✔ Saved colored mask: {output_colored}")


def main():
    p = argparse.ArgumentParser(description="Semantic Segmentation (HF Transformers) with labeled output.")
    p.add_argument("--input", required=True, help="Path to input image (PNG/JPG).")
    p.add_argument("--output", default="segmentation.png", help="Path to save the labeled overlay plot (PNG).")
    p.add_argument("--mask", default=None, help="Path to save raw class-ID mask (PNG). Default: <output_stem>_mask.png")
    p.add_argument("--colored", default=None, help="Optional: path to save colored mask image (PNG).")
    p.add_argument("--model", default="nvidia/segformer-b0-finetuned-ade-512-512",
                   help="HF model repo id for semantic segmentation.")
    p.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha ∈ [0,1].")
    p.add_argument("--max-legend", type=int, default=25, help="Max classes to list in the legend.")
    p.add_argument("--device", default="", help='Set to "cuda" or "cpu". Default: auto.')
    p.add_argument("--no-legend", action="store_true", help="Disable legend on the plot.")
    args = p.parse_args()

    output_plot = args.output
    if args.mask is None:
        stem = Path(output_plot).with_suffix("")
        output_mask = f"{stem}_mask.png"
    else:
        output_mask = args.mask

    run_segmentation(
        input_path=args.input,
        output_plot=output_plot,
        output_mask=output_mask,
        output_colored=args.colored,
        model_name=args.model,
        alpha=args.alpha,
        max_legend=args.max_legend,
        device=args.device,
        no_legend=args.no_legend,
    )


if __name__ == "__main__":
    main()

