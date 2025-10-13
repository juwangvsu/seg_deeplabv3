"""
Inference for PVTv2 + SegHeadLite. Supports YAML via --config.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm
from pvt_v2 import SegModel
from torchvision import transforms
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
CITYSCAPES_TRAINID_TO_COLOR = [
    (128, 64,128),(244, 35,232),( 70, 70, 70),(102,102,156),(190,153,153),
    (153,153,153),(250,170, 30),(220,220,  0),(107,142, 35),(152,251,152),
    ( 70,130,180),(220, 20, 60),(255,  0,  0),(  0,  0,142),(  0,  0, 70),
    (  0, 60,100),(  0, 80,100),(  0,  0,230),(119, 11, 32),
]
def default_palette(n:int):
    rng=np.random.RandomState(123)
    cols=[(int(rng.randint(0,255)),int(rng.randint(0,255)),int(rng.randint(0,255))) for _ in range(n)]
    if n>0: cols[0]=(0,128,255)
    return cols
def get_palette(mode: str, n: int):
    if mode == "cityscapes" or (mode == "auto" and n == 19):
        pal = CITYSCAPES_TRAINID_TO_COLOR.copy()
        if n <= len(pal): return pal[:n]
        return pal + [pal[-1]]*(n-len(pal))
    return default_palette(n)

def colorize_mask(mask_np: np.ndarray, palette) -> Image.Image:
    H,W = mask_np.shape
    rgba = np.zeros((H,W,4), dtype=np.uint8)
    for cls, col in enumerate(palette):
        m = mask_np == cls
        rgba[m,0], rgba[m,1], rgba[m,2], rgba[m,3] = col[0], col[1], col[2], 160
    return Image.fromarray(rgba, "RGBA")

def overlay_image(img: Image.Image, cm: Image.Image): return Image.alpha_composite(img.convert("RGBA"), cm)

IMG_EXTS={".jpg",".jpeg",".png",".bmp"}
class ImageFolder(Dataset):
    def __init__(self, images: List[Path], size: Tuple[int,int]):
        self.paths=images; self.size=size; self.to_tensor=T.ToTensor(); self.norm=T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        p=self.paths[i]; img=Image.open(p).convert("RGB"); orig=img.copy()
        #print('yyy self.size[::-1]', self.size[::-1])
        img=img.resize(self.size[::-1], Image.BILINEAR); t=self.norm(self.to_tensor(img))
        #print('yyy orig size', orig.size)
        return t, self.to_tensor(orig), p.name

def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    with open(path,"r") as f:
        import yaml as _yaml
        cfg=_yaml.safe_load(f) or {}
    if "infer" in cfg and isinstance(cfg["infer"], dict):
        base = cfg.get("base", {}); return {**base, **cfg["infer"]}
    return cfg

def build_argparser():
    ap=argparse.ArgumentParser(description="Inference for PVTv2 + SegHeadLite")
    ap.add_argument("--config", type=str, default=None)
    src=ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--images-dir", type=str)
    src.add_argument("--images-list", type=str)
    ap.add_argument("--load", type=str)
    ap.add_argument("--encoder", type=str, default="b2", choices=["b0","b1","b2","b3","b4","b5"])
    ap.add_argument("--num-classes", type=int, default=19)
    ap.add_argument("--size", type=int, nargs=2, default=[512,896], metavar=("H","W"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out-dir", type=str, default="runs/infer")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--palette", type=str, default="auto", choices=["auto","cityscapes","default"], help="Color palette for visualization")
    ap.add_argument('--showmodel', action='store_true')
    return ap

@torch.no_grad()
def main():
    base_ap=build_argparser(); pre,_=base_ap.parse_known_args(); cfg=load_yaml(pre.config)
    if cfg: base_ap.set_defaults(**cfg)
    args=base_ap.parse_args()

    paths: List[Path] = []
    if args.images_dir:
        for p in sorted(Path(args.images_dir).rglob("*")):
            if p.suffix.lower() in IMG_EXTS: paths.append(p)
    elif args.images_list:
        with open(args.images_list,"r") as f:
            for line in f:
                p=Path(line.strip())
                if p.suffix.lower() in IMG_EXTS and p.exists(): paths.append(p)
    if not paths: raise SystemExit("No images found; provide --images-dir or --images-list.")

    out_dir=Path(args.out_dir); masks_dir=out_dir/"masks"; color_dir=out_dir/"color_mask"; overlay_dir=out_dir/"overlay"
    for d in (masks_dir,color_dir,overlay_dir): ensure_dir(d)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=SegModel(variant=args.encoder, num_classes=args.num_classes).to(device)

    if args.load:
        ckpt=torch.load(args.load, map_location="cpu")
        meta=ckpt.get("meta", {})
        if "num_classes" in meta and meta["num_classes"] != args.num_classes:
            print(f"[warn] num_classes mismatch: ckpt={meta['num_classes']} vs arg={args.num_classes}")
        model.load_state_dict(ckpt.get("model", ckpt))
        #model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        print(f"Loaded weights from {args.load}")

    print(model)
    if args.showmodel:
        exit(0)
    palette = get_palette(args.palette, args.num_classes)

    model.eval()
    ds=ImageFolder(paths, size=(args.size[0], args.size[1]))
    dl=DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    for imgs, origs, names in tqdm(dl, desc="infer"):
        imgs=imgs.to(device, non_blocking=True); logits=model(imgs); preds=logits.argmax(1).cpu().numpy()
        for i in range(preds.shape[0]):
            name=names[i].rsplit('.',1)[0]
            mask=preds[i].astype(np.uint8)
            Image.fromarray(mask,"L").save(masks_dir/f"{name}.png")
            color=colorize_mask(mask, palette); color.save(color_dir/f"{name}.png")
            #print('xxx origs.shape ', origs.shape)
            if args.overlay:
                orig1=origs[i]
                to_pil = transforms.ToPILImage()
                orig = to_pil(orig1)
                #print('xxx color.size ', color.size)
                #print('xxx orig1.shape ', orig1.shape)
                #orig = Image.fromarray(origs[i].numpy()) #origs is a tensor
                #print('xxx orig.shape ', orig.size)
                over=overlay_image(orig, color.resize(orig.size, Image.NEAREST)); over.save(overlay_dir/f"{name}.png")

if __name__=="__main__": main()
