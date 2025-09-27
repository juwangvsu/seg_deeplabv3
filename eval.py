import os, argparse
import torch
import torch.nn.functional as F
from utils.train_utils import load_config, make_dataloaders
from models.deeplab import create_deeplabv3_resnet50
from utils.metrics import compute_mIoU, pixel_accuracy
from tqdm import tqdm

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader = make_dataloaders(cfg, input_size=None, eval_resize_long=cfg["eval"]["input_size"], shuffle=False)

    model = create_deeplabv3_resnet50(
        num_classes=cfg["model"]["num_classes"],
        pretrained_backbone=False
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    import numpy as np
    import collections
    stats = collections.defaultdict(float)
    count = 0

    for batch in tqdm(val_loader, desc="eval"):
        imgs = torch.from_numpy(batch["image"]).to(device)
        masks = torch.from_numpy(batch["mask"]).to(device)
        out = model(imgs)
        preds = torch.argmax(out["out"], dim=1)
        miou, per_cls = compute_mIoU(preds, masks, num_classes=cfg["model"]["num_classes"], ignore_index=cfg["train"]["ignore_index"])
        pixacc = pixel_accuracy(preds, masks, ignore_index=cfg["train"]["ignore_index"])
        stats["miou"] += float(miou)
        stats["pixacc"] += float(pixacc)
        count += 1

    for k in stats: stats[k] /= max(count,1)
    print(f"Eval: mIoU={stats['miou']:.3f} PixAcc={stats['pixacc']:.3f}")

if __name__ == "__main__":
    main()
