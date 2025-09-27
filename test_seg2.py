#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

def run_segmentation(input_path: str, output_path: str, show_plot: bool = False):
    # Load model + processor (ADE20K-trained SegFormer here)
    model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

    img = Image.open(input_path).convert("RGB")

    # Preprocess + forward pass
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    # Upsample to input image size
    logits = outputs.logits
    upsampled = torch.nn.functional.interpolate(
        logits,
        size=img.size[::-1],  # (H, W)
        mode="bilinear",
        align_corners=False,
    )
    pred = upsampled.argmax(1)[0].cpu()

    # Save result
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred, cmap="nipy_spectral")
    plt.title("Semantic Segmentation")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"[✔] Saved segmentation plot to {output_path}")

    if show_plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Segmentation Inference")
    parser.add_argument("--input", required=True, help="Path to input image (PNG/JPG)")
    parser.add_argument("--output", default="segmentation.png", help="Path to save output plot")
    parser.add_argument("--plot", action="store_true", help="Also display the plot interactively")

    args = parser.parse_args()
    run_segmentation(args.input, args.output, show_plot=args.plot)

