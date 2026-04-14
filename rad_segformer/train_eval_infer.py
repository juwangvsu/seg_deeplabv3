
# train_eval_infer.py
import os
import argparse
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from radar_dataset import RadarSegDataset, fast_hist, compute_miou_from_hist, pixel_accuracy
from model_design import RadarSegFormer
from utils_io import ensure_dir, save_checkpoint, load_checkpoint, save_pred_images

def get_args():
    p = argparse.ArgumentParser(description='Radar SegFormer Training / Eval / Infer')
    p.add_argument('--data-dir', type=str, required=True)
    p.add_argument('--radar-subdir', type=str, default='angle_range_numpy')
    p.add_argument('--out-dir', type=str, default='outputs')
    p.add_argument('--ckpt-dir', type=str, default='checkpoints')
    p.add_argument('--load', type=str, default=None, help='Checkpoint filename inside ckpt-dir (e.g., best.pt or epoch_0009.pt)')
    p.add_argument('--save-every', type=int, default=1, help='Save checkpoint every N epochs')

    p.add_argument('--mode', type=str, choices=['train', 'eval', 'infer'], default='train')
    p.add_argument('--input-file', type=str, default=None, help='For infer mode: a specific .npy file to run; default: all in data-dir')
    p.add_argument('--save-output', action='store_true', help='When eval/infer: save predicted PNGs and overlays')

    p.add_argument('--num-classes', type=int, required=True)
    p.add_argument('--variant', type=str, default='b2')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--image-size', type=int, default=512)
    p.add_argument('--val-split', type=float, default=0.1)

    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    return p.parse_args()

def make_loaders(data_dir: str, radar_subdir:str, image_size: int, batch_size: int, num_workers: int, val_split: float):
    ds = RadarSegDataset(data_dir=data_dir, radar_subdir=radar_subdir, image_size=(image_size, image_size))
    n_val = max(1, int(len(ds) * val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_ld = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_ld, val_ld

def evaluate(model: torch.nn.Module, loader: DataLoader, device: str, num_classes: int,
             save_output: bool = False, out_dir: Optional[str] = None):
    model.eval()
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in loader:
            imgs = batch['radar'].to(device)
            gts = batch['mask'].to(device).long()
            logits = model(imgs)
            # Resize logits to GT
            logits = F.interpolate(logits, size=gts.shape[-2:], mode='bilinear', align_corners=False)
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
            gts_np = gts.cpu().numpy().astype(np.uint8)

            for i in range(preds.shape[0]):
                hist += fast_hist(preds[i], gts_np[i], num_classes)

            if save_output and out_dir is not None:
                for i in range(preds.shape[0]):
                    stem = batch['stem'][i]
                    rgb_path = batch['rgb_path'][i]
                    save_pred_images(out_dir, stem, preds[i], rgb_path)

    miou = compute_miou_from_hist(hist)
    acc = pixel_accuracy(hist)
    return miou, acc

def train(args):
    device = args.device
    train_ld, val_ld = make_loaders(args.data_dir, args.radar_subdir, args.image_size, args.batch_size, args.num_workers, args.val_split)

    model = RadarSegFormer(num_classes=args.num_classes, variant=args.variant, image_size=args.image_size)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 0

    if args.load:
        start_epoch = load_checkpoint(args.ckpt_dir, args.load, model, optimizer)
        print(f"Loaded checkpoint '{args.load}' at epoch {start_epoch}")

    best_miou = -1.0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        for batch in train_ld:
            imgs = batch['radar'].to(device)
            gts = batch['mask'].to(device).long()

            logits = model(imgs)
            logits = torch.nn.functional.interpolate(logits, size=gts.shape[-2:], mode='bilinear', align_corners=False)

            loss = torch.nn.functional.cross_entropy(logits, gts, ignore_index=255)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        # Eval each epoch
        miou, acc = evaluate(model, val_ld, device, args.num_classes, save_output=args.save_output,
                              out_dir=os.path.join(args.out_dir, 'val_outputs'))
        print(f"Epoch {epoch+1}/{args.epochs} | Val mIoU={miou:.4f} Acc={acc:.4f}")

        # Save checkpoints
        is_best = miou > best_miou
        if is_best:
            best_miou = miou
        if ((epoch + 1) % args.save_every == 0) or is_best:
            save_checkpoint(args.ckpt_dir, epoch + 1, model, optimizer, best=is_best, best_metric=best_miou)

def run_eval(args):
    device = args.device
    _, val_ld = make_loaders(args.data_dir, args.image_size, args.batch_size, args.num_workers, args.val_split)

    model = RadarSegFormer(num_classes=args.num_classes, variant=args.variant, image_size=args.image_size)
    model.to(device)

    if args.load:
        load_checkpoint(args.ckpt_dir, args.load, model, optimizer=None)
        print(f"Loaded checkpoint '{args.load}'")
    else:
        print("Warning: running eval with randomly initialized weights.")

    miou, acc = evaluate(model, val_ld, device, args.num_classes, save_output=args.save_output,
                          out_dir=os.path.join(args.out_dir, 'eval_outputs'))
    print(f"Eval mIoU={miou:.4f} Acc={acc:.4f}")

def run_infer(args):
    device = args.device
    model = RadarSegFormer(num_classes=args.num_classes, variant=args.variant, image_size=args.image_size)
    model.to(device)

    if args.load:
        load_checkpoint(args.ckpt_dir, args.load, model, optimizer=None)
        print(f"Loaded checkpoint '{args.load}'")
    else:
        print("Warning: running inference with randomly initialized weights.")

    # Build a tiny dataset for either a single file or all files
    from radar_dataset import RadarSegDataset
    if args.input_file is not None:
        import os
        stem = os.path.splitext(os.path.basename(args.input_file))[0]
        ds = RadarSegDataset(args.data_dir, args.radar_subdir, image_size=(args.image_size, args.image_size), file_stems=[stem])
    else:
        ds = RadarSegDataset(args.data_dir, args.radar_subdir, image_size=(args.image_size, args.image_size))

    ld = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)

    model.eval()
    out_dir = os.path.join(args.out_dir, 'infer_outputs')
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for batch in ld:
            imgs = batch['radar'].to(device)
            logits = model(imgs)
            logits = torch.nn.functional.interpolate(logits, size=batch['mask'].shape[-2:], mode='bilinear', align_corners=False)
            preds = logits.argmax(dim=1).cpu().numpy()[0].astype(np.uint8)
            stem = batch['stem'][0]
            save_pred_images(out_dir, stem, preds, batch['rgb_path'][0])
            print(f"Saved outputs for {stem}")

if __name__ == '__main__':
    args = get_args()
    ensure_dir(args.out_dir)
    ensure_dir(args.ckpt_dir)

    if args.mode == 'train':
        train(args)
    elif args.mode == 'eval':
        run_eval(args)
    elif args.mode == 'infer':
        run_infer(args)
