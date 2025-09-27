import numpy as np
import torch

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n**2).reshape(n, n)

def compute_mIoU(pred, target, num_classes, ignore_index=255):
    # pred: [B, H, W], target: [B, H, W]
    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    hist = np.zeros((num_classes, num_classes))
    for p, t in zip(pred, target):
        t = t.copy()
        t[t == ignore_index] = -1
        hist += fast_hist(t.flatten(), p.flatten(), num_classes)
    iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)
    mIoU = np.nanmean(iu)
    return mIoU, iu

def pixel_accuracy(pred, target, ignore_index=255):
    pred = pred.detach().cpu()
    target = target.detach().cpu()
    mask = target != ignore_index
    correct = (pred[mask] == target[mask]).sum().item()
    total = mask.sum().item()
    return correct / max(total, 1)
