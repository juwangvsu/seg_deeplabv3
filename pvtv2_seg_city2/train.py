"""
Train PVTv2 + SegHeadLite with optional YAML config (--config) and Cityscapes support.
"""
from __future__ import annotations
import argparse, os, random
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
from PIL import Image
import yaml

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from pvt_v2 import SegModel

# -------------------------
# Utils
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
def ensure_dir(p: Path): p.mkdir(parents=True, exist_ok=True)
def default_palette(n: int):
    rng = np.random.RandomState(123)
    cols = [(int(rng.randint(0,255)),int(rng.randint(0,255)),int(rng.randint(0,255))) for _ in range(n)]
    if n>0: cols[0]=(0,128,255)
    return cols
def colorize_mask(mask_np: np.ndarray, n: int, ignore: int=255):
    H,W = mask_np.shape; rgba = np.zeros((H,W,4), dtype=np.uint8); pal=default_palette(n)
    for c in range(n): m = mask_np==c; rgba[m,0]=pal[c][0]; rgba[m,1]=pal[c][1]; rgba[m,2]=pal[c][2]; rgba[m,3]=120
    rgba[mask_np==ignore,3]=0; return Image.fromarray(rgba, "RGBA")
def overlay_image(img: Image.Image, cm: Image.Image): return Image.alpha_composite(img.convert("RGBA"), cm)

# -------------------------
# Datasets
# -------------------------
IMG_EXTS={".jpg",".jpeg",".png",".bmp"}; MASK_EXTS={".png",".bmp"}

class SegFolder(Dataset):
    """Generic folder dataset (DATA_ROOT/{train,val}/{images,masks})."""
    def __init__(self, root: str, split: str = "train", size: Tuple[int,int]=(512,896), augment: bool=False, ignore_index: int=255):
        self.root = Path(root); self.split = split; self.size = size; self.augment = augment; self.ignore_index = ignore_index
        imgd=self.root/split/"images"; maskd=self.root/split/"masks"
        assert imgd.is_dir() and maskd.is_dir(), f"Expected {imgd} and {maskd}"
        self.items: List[Tuple[Path,Path]] = []
        for p in sorted(imgd.rglob("*")):
            if p.suffix.lower() in IMG_EXTS:
                stem=p.stem
                cands=[c for c in maskd.rglob(stem+".*") if c.suffix.lower() in MASK_EXTS]
                if cands: self.items.append((p,cands[0]))
        if not self.items: raise RuntimeError("No image/mask pairs found.")
        self.to_tensor=T.ToTensor(); self.norm=T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    def __len__(self): return len(self.items)
    def __getitem__(self, idx: int):
        ip, mp = self.items[idx]
        img=Image.open(ip).convert("RGB"); mask=Image.open(mp)
        img=img.resize(self.size[::-1], Image.BILINEAR); mask=mask.resize(self.size[::-1], Image.NEAREST)
        if self.augment and random.random()<0.5: img=img.transpose(Image.FLIP_LEFT_RIGHT); mask=mask.transpose(Image.FLIP_LEFT_RIGHT)
        t=self.norm(self.to_tensor(img)); m_np=np.array(mask, dtype=np.int64); return t, torch.from_numpy(m_np), ip.name

CITYSCAPES_LABELIDS_TO_TRAINIDS = {7:0,8:1,11:2,12:3,13:4,17:5,19:6,20:7,21:8,22:9,23:10,24:11,25:12,26:13,27:14,28:15,31:16,32:17,33:18}
def _labelids_to_trainids(arr: np.ndarray, ignore_index: int=255) -> np.ndarray:
    out = np.full_like(arr, ignore_index)
    for k,v in CITYSCAPES_LABELIDS_TO_TRAINIDS.items(): out[arr==k]=v
    return out

class CityscapesSeg(Dataset):
    """Cityscapes official layout under data-root."""
    def __init__(self, root: str, split: str="train", size: Tuple[int,int]=(512,1024), augment: bool=False, ignore_index: int=255, mode: str="auto"):
        self.root=Path(root); self.split=split; self.size=size; self.augment=augment; self.ignore_index=ignore_index; self.mode=mode
        imgd=self.root/"leftImg8bit"/split; lbld=self.root/"gtFine"/split
        assert imgd.is_dir() and lbld.is_dir(), f"Missing Cityscapes dirs {imgd} / {lbld}"
        self.items: List[Tuple[Path,Path,str]] = []
        for p in sorted(imgd.rglob("*_leftImg8bit.png")):
            city=p.parent.name; stem=p.stem.replace("_leftImg8bit","")
            tid=lbld/city/f"{stem}_gtFine_labelTrainIds.png"
            lid=lbld/city/f"{stem}_gtFine_labelIds.png"
            chosen=None; mtype=None
            if self.mode=="trainIds":
                if tid.exists(): chosen, mtype = tid, "trainIds"
                else: continue
            elif self.mode=="labelIds":
                if lid.exists(): chosen, mtype = lid, "labelIds"
                else: continue
            else: # auto
                if tid.exists(): chosen, mtype = tid, "trainIds"
                elif lid.exists(): chosen, mtype = lid, "labelIds"
            if chosen is not None: self.items.append((p, chosen, mtype))
        if not self.items: raise RuntimeError("No Cityscapes pairs found.")
        self.to_tensor=T.ToTensor(); self.norm=T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    def __len__(self): return len(self.items)
    def __getitem__(self, idx: int):
        ip, mp, mtype = self.items[idx]
        img=Image.open(ip).convert("RGB"); mask=Image.open(mp)
        img=img.resize(self.size[::-1], Image.BILINEAR); mask=mask.resize(self.size[::-1], Image.NEAREST)
        if self.augment and random.random()<0.5: img=img.transpose(Image.FLIP_LEFT_RIGHT); mask=mask.transpose(Image.FLIP_LEFT_RIGHT)
        t=self.norm(self.to_tensor(img))
        m_np=np.array(mask, dtype=np.int64)
        if mtype=="labelIds": m_np=_labelids_to_trainids(m_np, ignore_index=self.ignore_index)
        return t, torch.from_numpy(m_np), ip.name

class StreamingConfusion:
    def __init__(self, n:int, ignore:int=255):
        self.n=n; self.ignore=ignore; self.mat=torch.zeros((n,n), dtype=torch.int64)
    @torch.no_grad()
    def update(self, tgt: torch.Tensor, pred: torch.Tensor):
        mask = tgt!=self.ignore; t=tgt[mask].view(-1); p=pred[mask].view(-1)
        k=(t*self.n + p).to(torch.int64); bc=torch.bincount(k, minlength=self.n*self.n)
        self.mat += bc.view(self.n,self.n)
    def compute(self):
        h=self.mat.float(); acc=torch.diag(h).sum()/h.sum().clamp(min=1.0)
        iu=torch.diag(h)/(h.sum(1)+h.sum(0)-torch.diag(h)).clamp(min=1.0)
        return acc.item(), iu.mean().item(), iu.tolist()

@torch.no_grad()
def evaluate(model, loader, device, ncls, out_dir: Optional[Path], save_preds: bool, ignore_index: int):
    model.eval(); conf=StreamingConfusion(ncls, ignore_index)
    masks_dir=color_dir=overlay_dir=None
    if save_preds and out_dir is not None:
        masks_dir=out_dir/"masks"; color_dir=out_dir/"color_mask"; overlay_dir=out_dir/"overlay"
        for d in (masks_dir,color_dir,overlay_dir): ensure_dir(d)
    for imgs, masks, names in tqdm(loader, desc="eval", leave=False):
        imgs=imgs.to(device, non_blocking=True)
        logits=model(imgs); preds=logits.argmax(1).cpu()
        for i in range(imgs.size(0)):
            gt=masks[i].cpu(); pd=preds[i]; conf.update(gt,pd)
            if save_preds and out_dir is not None:
                Image.fromarray(pd.numpy().astype(np.uint8),"L").save(masks_dir/f"{names[i].rsplit('.',1)[0]}.png")
                color=colorize_mask(pd.numpy(), ncls, ignore_index); color.save(color_dir/f"{names[i].rsplit('.',1)[0]}.png")
                img_np=(imgs[i].cpu().numpy().transpose(1,2,0)*np.array([0.229,0.224,0.225])+np.array([0.485,0.456,0.406]))
                img_np=np.clip(img_np*255.0,0,255).astype(np.uint8); overlay=overlay_image(Image.fromarray(img_np), color); overlay.save(overlay_dir/f"{names[i].rsplit('.',1)[0]}.png")
    return conf.compute()

def train_one_epoch(model, loader, optim, scaler, device, criterion, max_norm: float):
    model.train(); run=0.0; n=0; pbar=tqdm(loader, desc="train", leave=False)
    for imgs, masks, _ in pbar:
        imgs=imgs.to(device, non_blocking=True); masks=masks.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits=model(imgs); loss=criterion(logits, masks)
        scaler.scale(loss).backward()
        if max_norm and max_norm>0:
            scaler.unscale_(optim); torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optim); scaler.update()
        run += loss.item()*imgs.size(0); n += imgs.size(0); pbar.set_postfix(loss=run/max(1,n))
    return run/max(1,n)

def load_yaml(path: Optional[str]) -> Dict[str, Any]:
    if not path: return {}
    with open(path,"r") as f:
        import yaml as _yaml
        cfg=_yaml.safe_load(f) or {}
    if "train" in cfg and isinstance(cfg["train"], dict):
        base = cfg.get("base", {}); return {**base, **cfg["train"]}
    return cfg

def build_argparser():
    ap=argparse.ArgumentParser(description="Train PVTv2 + SegHeadLite")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--data-root", type=str)
    ap.add_argument("--dataset", type=str, default="folder", choices=["folder","cityscapes"], help="Dataset type")
    ap.add_argument("--cs-mode", type=str, default="auto", choices=["auto","labelIds","trainIds"], help="Cityscapes mask handling")
    ap.add_argument("--encoder", type=str, default="b2", choices=["b0","b1","b2","b3","b4","b5"])
    ap.add_argument("--num-classes", type=int, default=19)
    ap.add_argument("--size", type=int, nargs=2, default=[512,896], metavar=("H","W"))
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--ignore-index", type=int, default=255)
    ap.add_argument("--out-dir", type=str, default="runs/exp")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--save-val-pred", action="store_true")
    ap.add_argument("--load", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    return ap

def main():
    base_ap=build_argparser(); pre,_=base_ap.parse_known_args(); cfg=load_yaml(pre.config)
    if cfg: base_ap.set_defaults(**cfg)
    args=base_ap.parse_args()
    if not args.data_root: raise SystemExit("Provide --data-root (or via --config train.data-root)")
    set_seed(args.seed)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir=Path(args.out_dir); ensure_dir(out_dir)

    size=(args.size[0], args.size[1])
    if args.dataset=="cityscapes":
        train_set=CityscapesSeg(args.data_root,"train",size,True,args.ignore_index,args.cs_mode)
        val_set=CityscapesSeg(args.data_root,"val",size,False,args.ignore_index,args.cs_mode)
    else:
        train_set=SegFolder(args.data_root,"train",size,True,args.ignore_index)
        val_set=SegFolder(args.data_root,"val",size,False,args.ignore_index)

    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True,drop_last=True)
    val_loader=DataLoader(val_set,batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)

    model=SegModel(variant=args.encoder, num_classes=args.num_classes).to(device)
    criterion=nn.CrossEntropyLoss(ignore_index=args.ignore_index)
    optim=torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler=torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    start_epoch=0; best_miou=-1.0
    if args.load and os.path.isfile(args.load):
        ckpt=torch.load(args.load, map_location="cpu")
        model.load_state_dict(ckpt.get("model", ckpt), strict=False)
        if all(k in ckpt for k in ("optim","scaler","epoch")):
            optim.load_state_dict(ckpt["optim"]); scaler.load_state_dict(ckpt["scaler"]); start_epoch=int(ckpt["epoch"])+1
            best_miou=float(ckpt.get("best_miou", best_miou))
        print(f"Loaded checkpoint {args.load} (resume from epoch {start_epoch})")

    if args.eval:
        acc, miou, _ = evaluate(model, val_loader, device, args.num_classes, out_dir, True, args.ignore_index)
        print(f"Eval-only — Acc: {acc:.4f}  mIoU: {miou:.4f}"); return

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        tr_loss=train_one_epoch(model, train_loader, optim, scaler, device, criterion, args.max_grad_norm)
        acc, miou, _ = evaluate(model, val_loader, device, args.num_classes, out_dir if args.save_val_pred else None, args.save_val_pred, args.ignore_index)
        print(f"train_loss={tr_loss:.4f}  val_acc={acc:.4f}  val_mIoU={miou:.4f}")
        meta={"encoder":args.encoder,"num_classes":args.num_classes,"size":list(args.size),"ignore_index":args.ignore_index}
        if (epoch+1) % args.save_every == 0:
            path=out_dir/f"checkpoint_epoch{epoch+1:03d}.pth"
            torch.save({"epoch":epoch,"model":model.state_dict(),"optim":optim.state_dict(),"scaler":scaler.state_dict(),"best_miou":best_miou,"meta":meta}, path)
            print(f"Saved {path}")
        if miou > best_miou:
            best_miou=miou; best_path=out_dir/"best.pth"
            torch.save({"epoch":epoch,"model":model.state_dict(),"optim":optim.state_dict(),"scaler":scaler.state_dict(),"best_miou":best_miou,"meta":meta}, best_path)
            print(f"Saved best to {best_path} (mIoU={best_miou:.4f})")

if __name__=="__main__": main()
