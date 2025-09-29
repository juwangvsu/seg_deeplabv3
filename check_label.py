import torch, numpy as np, os
from utils.train_utils import load_config, make_dataloaders
cfg = load_config("configs/apgdata.yaml")
tr, _ = make_dataloaders(cfg, input_size=cfg["train"]["input_size"], eval_resize_long=None)
batch = next(iter(tr))
m = batch["mask"]
print("mask dtype:", m.dtype, "min:", m.min(), "max:", m.max())
vals = np.unique(m)
print("unique labels:", vals[:50], "count:", len(vals))
bad = vals[(vals != 255) & ((vals < 0) | (vals > cfg["model"]["num_classes"]-1))]
print("bad labels:", bad)
