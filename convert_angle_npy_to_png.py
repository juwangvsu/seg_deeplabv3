#!/usr/bin/env python3
"""
Convert an angle_range NumPy file (.npy) to a pseudo-colored PNG.

See --help for details and examples.
"""

import argparse, math, sys
from pathlib import Path
import numpy as np
try:
    import imageio.v3 as iio
except Exception:
    import imageio as iio
from matplotlib import cm, colors

def parse_args():
    p = argparse.ArgumentParser(description='Convert angle_range .npy arrays to pseudo-colored PNGs.')
    p.add_argument('-i','--input', required=True, type=str,
                   help='Path to a .npy file or a directory containing .npy files.')
    p.add_argument('-o','--output', required=True, type=str,
                   help='Path to output file (.png) or directory (for batch).')
    p.add_argument('--cmap', default='hsv', type=str,
                   help='Matplotlib colormap name (e.g., hsv, turbo, viridis, jet). Default: hsv')
    p.add_argument('--vmin', type=float, default=None,
                   help='Lower value for normalization. Default: -pi (if not --autoscale).')
    p.add_argument('--vmax', type=float, default=None,
                   help='Upper value for normalization. Default:  pi (if not --autoscale).')
    p.add_argument('--autoscale', action='store_true',
                   help='Compute vmin/vmax from finite data.')
    p.add_argument('--clip', action='store_true',
                   help='Clip values outside [vmin, vmax].')
    p.add_argument('--nan-color', type=str, default=None,
                   help="Hex RGB for NaNs (e.g., FF00FF). Omit to keep NaNs transparent.")
    p.add_argument('--nan-alpha', type=int, default=0,
                   help='Alpha [0..255] for NaN pixels. Default: 0 (transparent).')
    p.add_argument('--grayscale16', action='store_true',
                   help='Save 16-bit grayscale PNG instead of color (scaled to [0, 65535]).')
    p.add_argument('--suffix', type=str, default='.png',
                   help='Output suffix/extension for batch mode. Default: .png')
    return p.parse_args()

def hex_to_rgb(s: str):
    s = s.strip().lstrip('#')
    if len(s) not in (3,6):
        raise ValueError('nan-color must be 3 or 6 hex digits')
    if len(s)==3:
        s = ''.join([c*2 for c in s])
    return int(s[0:2],16), int(s[2:4],16), int(s[4:6],16)

def compute_minmax(arr: np.ndarray):
    finite = np.isfinite(arr)
    if not finite.any():
        raise ValueError('No finite values found for autoscale.')
    data = arr[finite]
    return float(np.min(data)), float(np.max(data))

def normalize(arr: np.ndarray, vmin: float, vmax: float, clip: bool):
    if vmin >= vmax:
        raise ValueError(f'vmin ({vmin}) must be < vmax ({vmax})')
    if clip:
        norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
        return norm(arr)
    return (arr - vmin) / (vmax - vmin)

def to_color(arr, vmin, vmax, cmap_name, clip, nan_color=None, nan_alpha=0):
    normed = normalize(arr, vmin, vmax, clip=clip)
    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(normed, bytes=True)
    mask = ~np.isfinite(arr)
    if mask.any():
        if nan_color is None:
            rgba[mask,3] = 0
        else:
            r,g,b = nan_color
            rgba[mask,0] = r; rgba[mask,1] = g; rgba[mask,2] = b; rgba[mask,3] = np.uint8(np.clip(nan_alpha,0,255))
    return rgba

def to_u16(arr, vmin, vmax, clip):
    normed = normalize(arr, vmin, vmax, clip=clip)
    normed = np.clip(normed, 0.0, 1.0)
    u16 = (normed * 65535.0 + 0.5).astype(np.uint16)
    mask = ~np.isfinite(arr)
    if mask.any(): u16[mask] = 0
    return u16

def convert_file(in_path: Path, out_path: Path, *, cmap: str, vmin, vmax, autoscale: bool,
                 clip: bool, nan_color_hex: str | None, nan_alpha: int, grayscale16: bool):
    arr = np.load(in_path)
    if arr.ndim != 2:
        if arr.ndim == 3 and (arr.shape[-1]==1 or arr.shape[0]==1):
            arr = np.squeeze(arr)
        else:
            raise ValueError(f'Expected 2D array, got shape {arr.shape} in {in_path}')
    if autoscale:
        vmin_data, vmax_data = compute_minmax(arr)
        vmin = vmin_data if vmin is None else vmin
        vmax = vmax_data if vmax is None else vmax
    else:
        if vmin is None: vmin = -math.pi
        if vmax is None: vmax =  math.pi
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if grayscale16:
        img = to_u16(arr, vmin, vmax, clip=clip)
        iio.imwrite(out_path.as_posix(), img)
    else:
        nan_rgb = hex_to_rgb(nan_color_hex) if nan_color_hex else None
        rgba = to_color(arr, vmin, vmax, cmap, clip, nan_rgb, nan_alpha)
        iio.imwrite(out_path.as_posix(), rgba)
    return vmin, vmax

def main():
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    if in_path.is_dir():
        out_path.mkdir(parents=True, exist_ok=True)
        npy_files = sorted(in_path.glob('*.npy'))
        if not npy_files:
            print(f'No .npy files found in directory: {in_path}', file=sys.stderr); sys.exit(1)
        vmin, vmax = args.vmin, args.vmax
        if args.autoscale and (vmin is None or vmax is None):
            mins, maxs = [], []
            for f in npy_files:
                arr = np.load(f)
                mn, mx = compute_minmax(arr)
                mins.append(mn); maxs.append(mx)
            vmin = min(mins) if vmin is None else vmin
            vmax = max(maxs) if vmax is None else vmax
        for f in npy_files:
            out_f = out_path / (f.stem + args.suffix)
            vmin_used, vmax_used = convert_file(
                f, out_f, cmap=args.cmap, vmin=vmin, vmax=vmax,
                autoscale=False if (vmin is not None and vmax is not None) else args.autoscale,
                clip=args.clip, nan_color_hex=args.nan_color, nan_alpha=args.nan_alpha,
                grayscale16=args.grayscale16,
            )
            print(f'[OK] {f.name} → {out_f.name}  (vmin={vmin_used:.6f}, vmax={vmax_used:.6f})')
    else:
        if out_path.is_dir():
            out_path = out_path / (in_path.stem + args.suffix)
        vmin_used, vmax_used = convert_file(
            in_path, out_path, cmap=args.cmap, vmin=args.vmin, vmax=args.vmax,
            autoscale=args.autoscale, clip=args.clip, nan_color_hex=args.nan_color,
            nan_alpha=args.nan_alpha, grayscale16=args.grayscale16,
        )
        print(f'[OK] {in_path.name} → {out_path.name}  (vmin={vmin_used:.6f}, vmax={vmax_used:.6f})')

if __name__ == '__main__':
    main()
