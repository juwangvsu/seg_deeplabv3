import os
import argparse
import glob
import json
from typing import Dict, Tuple, List
import cv2
import numpy as np

DEFAULT_CLASSES = {
    0: "background",
    1: "pavement",
    2: "person",
    3: "car",
    4: "tree",
}

# BGR colors for OpenCV (so they match saves)
DEFAULT_COLORS = {
    0: (0, 0, 0),         # background - black
    1: (128, 128, 128),   # pavement - gray
    2: (0, 0, 255),       # person - red (BGR)
    3: (0, 255, 0),       # car - green
    4: (255, 0, 0),       # tree - blue
}

def make_legend(canvas: np.ndarray, class_names: Dict[int, str], colors: Dict[int, Tuple[int,int,int]]):
    pad = 8
    box = 16
    x, y = pad, pad
    for k in sorted(class_names.keys()):
        c = colors.get(k, (255,255,255))
        cv2.rectangle(canvas, (x, y), (x+box, y+box), c, -1)
        cv2.rectangle(canvas, (x, y), (x+box, y+box), (0,0,0), 1)
        text = f"{k}: {class_names[k]}"
        cv2.putText(canvas, text, (x+box+8, y+box-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(canvas, text, (x+box+8, y+box-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
        y += box + 6
    return canvas

def colorize_mask(mask: np.ndarray, colors: Dict[int, Tuple[int,int,int]], ignore_index: int = 255) -> np.ndarray:
    h, w = mask.shape[:2]
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for k, col in colors.items():
        color[mask == k] = col
    color[mask == ignore_index] = (0, 0, 0)  # keep ignore black
    return color

def overlay(img: np.ndarray, color_mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 1.0, color_mask, alpha, 0)

def vis_one(image_path: str, mask_path: str, out_path: str, class_names: Dict[int,str], colors: Dict[int,Tuple[int,int,int]], alpha: float, legend: bool):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    color_mask = colorize_mask(mask, colors)
    over = overlay(img, color_mask, alpha=alpha)

    if legend:
        legend_canvas = np.zeros((max(120, img.shape[0]//3), img.shape[1], 3), dtype=np.uint8)
        legend_canvas[:] = (240,240,240)
        legend_canvas = make_legend(legend_canvas, class_names, colors)
        over = np.vstack([over, legend_canvas])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, over)

def main():
    ap = argparse.ArgumentParser(description="Visualize segmentation masks over images with a color legend.")
    ap.add_argument("--images", type=str, required=True, help="Path to images directory")
    ap.add_argument("--masks", type=str, required=True, help="Path to masks directory (IDs)")
    ap.add_argument("--maskpref", type=str, default="", help="Path to masks directory (IDs)")
    ap.add_argument("--output_dir", type=str, required=True, help="Where to save overlays")
    ap.add_argument("--alpha", type=float, default=0.5, help="Mask overlay opacity")
    ap.add_argument("--legend", action="store_true", help="Append a legend below the image")
    ap.add_argument("--classes_json", type=str, default=None, help="Optional JSON {id: name} for classes")
    ap.add_argument("--colors_json", type=str, default=None, help="Optional JSON {id: [B,G,R]} for colors")
    ap.add_argument("--glob", type=str, default="*.*", help="Glob pattern for images (default '*.*')")
    args = ap.parse_args()

    class_names = DEFAULT_CLASSES.copy()
    colors = DEFAULT_COLORS.copy()

    if args.classes_json and os.path.isfile(args.classes_json):
        with open(args.classes_json, "r") as f:
            class_names = {int(k): v for k, v in json.load(f).items()}
    if args.colors_json and os.path.isfile(args.colors_json):
        with open(args.colors_json, "r") as f:
            raw = json.load(f)
            colors = {int(k): tuple(map(int, v)) for k, v in raw.items()}

    img_paths = sorted(glob.glob(os.path.join(args.images, args.glob)))
    if not img_paths:
        raise RuntimeError(f"No images found in {args.images} with pattern {args.glob}")

    for p in img_paths:
        name = os.path.basename(p)
        if args.maskpref=="":
            mask_p = os.path.join(args.masks, os.path.splitext(name)[0] + ".png")
        else:
            mask_p = os.path.join(args.masks, os.path.splitext(name)[0] +"_"+args.maskpref+ ".png")
        print('xxx mask fn ', mask_p)
        if not os.path.isfile(mask_p):
            print(f"[warn] Mask not found for {name}; skipping")
            continue
        out_p = os.path.join(args.output_dir, os.path.splitext(name)[0] + "_overlay.png")
        vis_one(p, mask_p, out_p, class_names, colors, alpha=args.alpha, legend=args.legend)

    print(f"Saved overlays to: {args.output_dir}")

if __name__ == "__main__":
    main()
