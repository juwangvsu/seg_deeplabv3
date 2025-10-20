#!/usr/bin/env python3
import os
import argparse
from PIL import Image
import numpy as np

def get_unique_ids(mask_path):
    """Return sorted unique pixel values from a segmentation mask."""
    mask = np.array(Image.open(mask_path))
    return np.unique(mask)

def main():
    parser = argparse.ArgumentParser(description="Print unique label IDs from segmentation PNG files.")
    parser.add_argument("--mask_dir", required=True, help="Path to folder containing segmentation PNG masks.")
    parser.add_argument("--recursive", action="store_true", help="Search masks recursively in subfolders.")
    parser.add_argument("--fnnotstrict", action="store_true", help="Search masks recursively in subfolders.")
    args = parser.parse_args()

    mask_dir = args.mask_dir
    all_ids = set()

    # Collect all .png files
    if args.recursive:
        if args.fnnotstrict:
            mask_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(mask_dir)
                for f in files if f.endswith(".png")
            ]
        else:
            mask_files = [
                os.path.join(root, f)
                for root, _, files in os.walk(mask_dir)
                for f in files if f.endswith("labelIds.png")
            ]

    else:
        mask_files = [os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.lower().endswith(".png")]

    if not mask_files:
        print(f"No PNG masks found in {mask_dir}")
        return

    print(f"Found {len(mask_files)} mask files. Scanning for unique IDs...")

    for path in mask_files:
        unique_ids = get_unique_ids(path)
        all_ids.update(unique_ids.tolist())

    all_ids = sorted(all_ids)
    print("\nUnique label IDs across all masks:")
    print(all_ids)
    print(f"\nTotal unique IDs: {len(all_ids)}")

if __name__ == "__main__":
    main()

