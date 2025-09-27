import os, math, time, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from utils.train_utils import load_config, make_dataloaders, save_checkpoint
from models.deeplab import create_deeplabv3_resnet50
from utils.metrics import compute_mIoU, pixel_accuracy
from tqdm import tqdm

def train_one_epoch(model, loader, optimizer, scaler, device, ignore_index):
    model.train()
    running = {"loss": 0.0, "pixacc": 0.0}
    for batch in tqdm(loader, desc="train", leave=False):
        #imgs = torch.from_numpy(batch["image"]).to(device)
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        #masks = torch.from_numpy(batch["mask"]).to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            out = model(imgs)
            logits = out["out"]  # [B, C, H, W]
            loss = F.cross_entropy(logits, masks.long(), ignore_index=ignore_index)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            pixacc = pixel_accuracy(preds, masks, ignore_index=ignore_index)
        running["loss"] += loss.item()
        running["pixacc"] += pixacc
    n = len(loader)
    return {k: v / max(n,1) for k,v in running.items()}

@torch.no_grad()
def evaluate(model, loader, device, num_classes, ignore_index, eval_resize_long):
    model.eval()
    from collections import defaultdict
    stats = defaultdict(float)
    count = 0
    for batch in tqdm(loader, desc="val", leave=False):
        imgs = batch["image"].to(device)
        masks = batch["mask"].to(device)
        out = model(imgs)
        logits = out["out"]
        preds = torch.argmax(logits, dim=1)
        miou, per_cls = compute_mIoU(preds, masks, num_classes=num_classes, ignore_index=ignore_index)
        pixacc = pixel_accuracy(preds, masks, ignore_index=ignore_index)
        stats["miou"] += float(miou)
        stats["pixacc"] += float(pixacc)
        count += 1
    for k in stats:
        stats[k] /= max(count,1)
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    out_dir = cfg["train"]["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 1337))

    # Data
    tr_loader, va_loader = make_dataloaders(
        cfg,
        input_size=cfg["train"]["input_size"],
        eval_resize_long=cfg["eval"]["input_size"],
    )

    # Model
    model = create_deeplabv3_resnet50(
        num_classes=cfg["model"]["num_classes"],
        pretrained_backbone=cfg["model"].get("pretrained_backbone", True)
    ).to(device)

    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    scaler = GradScaler(enabled=cfg["train"]["amp"])

    best_miou = -1.0
    latest_path = os.path.join(out_dir, "latest.ckpt")

    for epoch in range(1, cfg["train"]["max_epochs"]+1):
        tr_stats = train_one_epoch(model, tr_loader, opt, scaler, device, cfg["train"]["ignore_index"])
        va_stats = evaluate(model, va_loader, device, cfg["model"]["num_classes"], cfg["train"]["ignore_index"], cfg["eval"]["input_size"])
        print(f"[epoch {epoch:03d}] loss={tr_stats['loss']:.4f} pixacc={tr_stats['pixacc']:.3f} | val_mIoU={va_stats['miou']:.3f} val_pixacc={va_stats['pixacc']:.3f}")

        # Save latest
        save_checkpoint({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "val": va_stats}, latest_path)
        if epoch % cfg["train"]["save_every"] == 0:
            save_checkpoint({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "val": va_stats}, os.path.join(out_dir, f"epoch_{epoch:03d}.ckpt"))
        if va_stats["miou"] > best_miou:
            best_miou = va_stats["miou"]
            save_checkpoint({"epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "val": va_stats}, os.path.join(out_dir, "best.ckpt"))

if __name__ == "__main__":
    main()
