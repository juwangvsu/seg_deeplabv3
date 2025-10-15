from torch.utils.data import DataLoader, random_split
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from segformer import SegFormer
from data import SegDataset
from data_city import SegFolder # cityscape dataset
from utils import load_config, save_checkpoint

def compute_mIoU(conf_matrix: torch.Tensor) -> float:
    diag = torch.diag(conf_matrix)
    denom = conf_matrix.sum(1) + conf_matrix.sum(0) - diag
    iou = diag / torch.clamp(denom, min=1)
    return float((iou.mean().item()))

def update_confmat(conf, preds, targets, num_classes, ignore_index):
    mask = targets != ignore_index
    preds = preds[mask]
    targets = targets[mask]
    k = (targets >= 0) & (targets < num_classes)
    inds = num_classes * targets[k].to(torch.int64) + preds[k]
    conf += torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)

def load_model_weights(model: torch.nn.Module, path: str):
    sd = torch.load(path, map_location='cpu')
    state = None
    if isinstance(sd, dict) and 'model' in sd and isinstance(sd['model'], dict):
        state = sd['model']
    elif isinstance(sd, dict):
        state = sd
    else:
        raise RuntimeError(f"Unrecognized checkpoint format at {path}")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded weights from {path}\\n  missing keys: {len(missing)} | unexpected keys: {len(unexpected)}")
    if missing:
        print("  (missing) e.g.:", missing[:10])
    if unexpected:
        print("  (unexpected) e.g.:", unexpected[:10])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yml')
    parser.add_argument('--encoder', type=str, default='pvt_v2_b2', help="timm backbone (e.g., pvt_v2_b2, pvt_v2_b1, mit_b2, ...)")
    parser.add_argument('--load', type=str, default=None, help="Load model weights from .pth (no optimizer/scheduler state)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg['data']
    model_cfg = cfg['model']
    train_cfg = cfg['train']
    dataset_type = data_cfg['datatype']
    img_size = data_cfg['img_size']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if dataset_type=="cityscapes":
        ds = SegFolder(data_cfg['root'], image_size=img_size, aug=True, dataset=dataset_type)
        n_val = max(1, int(0.1 * len(ds)))
        n_train = len(ds) - n_val
        train_set, val_set = random_split(ds, [n_train, n_val])
    else:
        train_set = SegDataset(
            root=data_cfg['root'], split='', img_dirname=data_cfg['train_images'], mask_dirname=data_cfg['train_masks'],
            input_size=train_cfg['input_size'], ignore_index=train_cfg['ignore_index']
        )
        val_set = SegDataset(
            root=data_cfg['root'], split='', img_dirname=data_cfg['val_images'], mask_dirname=data_cfg['val_masks'],
            input_size=train_cfg['input_size'], ignore_index=train_cfg['ignore_index']
        )

    train_loader = DataLoader(train_set, batch_size=train_cfg['batch_size'], shuffle=True, num_workers=train_cfg['num_workers'], pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=train_cfg['batch_size'], shuffle=False, num_workers=train_cfg['num_workers'], pin_memory=True)

    model = SegFormer(num_classes=model_cfg['num_classes'], encoder_variant=args.encoder).to(device)

    # --load takes precedence over config['train']['resume'] for weights-only init
    if args.load:
        load_model_weights(model, args.load)

    criterion = nn.CrossEntropyLoss(ignore_index=train_cfg['ignore_index'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])
    total_iters = max(1, len(train_loader) * train_cfg['max_epochs'])
    scheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=total_iters, power=0.9)

    scaler = GradScaler(enabled=bool(train_cfg.get('amp', True)))

    start_epoch = 0

    # Resume full training state (optimizer/scheduler/scaler + epoch). Only if --load not provided.
    if (not args.load) and train_cfg.get('resume'):
        ckpt = torch.load(train_cfg['resume'], map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from {train_cfg['resume']} at epoch {start_epoch}")

    os.makedirs(train_cfg['output_dir'], exist_ok=True)
    save_every = int(train_cfg.get('save_every', 5))

    for epoch in range(start_epoch, train_cfg['max_epochs']):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['max_epochs']} [train]")
        for batch in pbar:
            imgs = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                logits = model(imgs)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        avg_loss = running_loss / max(1, len(train_loader))

        model.eval()
        conf = torch.zeros(model_cfg['num_classes'], model_cfg['num_classes'], dtype=torch.int64)
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{train_cfg['max_epochs']} [val]")
            for batch in pbar_val:
                imgs = batch['image'].to(device, non_blocking=True)
                masks = batch['mask'].to(device, non_blocking=True)
                with autocast(enabled=scaler.is_enabled()):
                    logits = model(imgs)
                    loss = criterion(logits, masks)
                val_loss += loss.item()
                preds = logits.argmax(dim=1).cpu()
                update_confmat(conf, preds, masks.cpu(), model_cfg['num_classes'], train_cfg['ignore_index'])
                pbar_val.set_postfix(loss=f"{loss.item():.4f}")

        miou = compute_mIoU(conf)
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch+1}/{train_cfg['max_epochs']} | train_loss={avg_loss:.4f} | val_loss={val_loss:.4f} | mIoU={miou:.4f}")

        # Save strictly every N epochs (no best-mIoU saving)
        if ((epoch + 1) % save_every) == 0:
            ckpt_path = os.path.join(train_cfg['output_dir'], f"epoch_{epoch+1:03d}.pth")
            save_checkpoint({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

if __name__ == '__main__':
    main()
