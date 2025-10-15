import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from backbones import create_pyramid_backbone
from segformer import SegFormer
from data import SegFolder

def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> float:
    #print('xxx pred target shape ', pred.shape, target.shape)
    pred = pred.view(-1)
    target = target.view(-1)
    #print('xxx pred target shape ', pred.shape, target.shape, torch.unique(pred), torch.unique(target))
    mask = target >= 0
    mask = target != 255
    pred = pred[mask]
    target = target[mask]
    hist = torch.bincount((target * num_classes + pred).to(torch.int64), minlength=num_classes*num_classes
        ).reshape(num_classes, num_classes).float()
    iou = torch.diag(hist) / (hist.sum(1) + hist.sum(0) - torch.diag(hist) + 1e-6)
    return float(iou.mean().item())

def train_one_epoch(model, loader, optimizer, scaler, device, criterion, amp=True):
    model.train()
    running = 0.0
    pbar = tqdm(loader, desc="train", ncols=100)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            logits = model(imgs)
            loss = criterion(logits, masks)

        if amp:
            # ✅ Correct AMP pattern
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # CPU / AMP disabled
            loss.backward()
            optimizer.step()
        running += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=f"{loss.item():.3f}")
    return running / len(loader.dataset)

def evaluate(model, loader, device, num_classes):
    model.eval()
    miou = 0.0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            pred = logits.argmax(1)
            miou += compute_miou(pred.cpu(), masks.cpu(), num_classes)
    return miou / max(1, len(loader))

def cross_entropy_2d(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255):
    return nn.functional.cross_entropy(logits, target, ignore_index=ignore_index)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="mit_b2")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--img_size", type=int, nargs=2, default=[512,896], metavar=("H","W"))

    parser.add_argument("--out", type=str, default="runs/exp1")
    parser.add_argument("--dstype", type=str, default="folder") #or cityscapes
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--boverride", type=str, default="")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = SegFolder(args.data_root, image_size=args.img_size, aug=True, dataset=args.dstype)
    n_val = max(1, int(0.1 * len(ds)))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    bk, channels, _ = create_pyramid_backbone(args.backbone, pretrained=True)
    model = SegFormer(bk, channels, num_classes=args.num_classes, decoder_embed_dim=256)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=(not args.no_amp) and device.type == "cuda")

    criterion = cross_entropy_2d

    start_epoch = 0
    if args.load:
        ckpt = torch.load(args.load, map_location="cpu")
        #resume ckpt must be a full model
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        if "optim" in ckpt:
            optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Resumed from {args.load} at epoch {start_epoch}")

    #load backbone if ...
    #model.backbone.state_dict().keys()
    if not args.boverride=="":
        print(f"override backbone parameter")
        #model.backbone.state_dict().keys()
        bbckpt = torch.load(args.boverride, map_location="cpu")
        #print('xxx bbchpt.keys', bbckpt.keys())
        model.backbone.load_state_dict(bbckpt)
    #print('xxx model.backbone.keys', model.backbone.state_dict().keys())
    #exit(0)
    best_miou = 0.0
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, device, criterion, amp=not args.no_amp)
        miou = evaluate(model, val_loader, device, args.num_classes)
        print(f"Epoch {epoch+1}/{args.epochs} | loss {train_loss:.4f} | mIoU {miou:.4f}")

        ckpt_path = out / f"epoch_{epoch+1}.pth"
        torch.save({"model": model.state_dict(), "optim": optimizer.state_dict(), "epoch": epoch+1}, ckpt_path)
        if miou > best_miou:
            best_miou = miou
            torch.save({"model": model.state_dict()}, out / "best.pth")

if __name__ == "__main__":
    main()
