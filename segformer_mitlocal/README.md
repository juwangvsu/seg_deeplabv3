# SegFormer (minimal) with local MiT backbone + HF weights loader

This package provides:
- `segformer.py`: SegFormer MLP decoder head
- `mit.py`: MixVisionTransformer (MiT) backbone (B0..B5)
- `backbones.py`: unified loader (timm backbones or local MiT via `mit_b*`)
- `data.py`: simple folder dataset loader
- `palettes.py`: Cityscapes palette
- `train.py`: AMP training with checkpointing + mIoU
- `infer.py`: inference (masks, color, overlay)
- `scripts/load_from_hf.py`: **convert** Hugging Face SegFormer encoder weights into this local `mit.py` format

## Install
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data structure (example)
```
DATA_ROOT/
  images/*.jpg|png
  masks/*.png     # class IDs [0..C-1]
```

## Train (MiT backbone from scratch)
```bash
python train.py --data-root /path/to/DATA_ROOT --backbone mit_b2 --num-classes 19 --epochs 40 --batch-size 6 --img-size 768 --out runs/exp_mit
```

## Convert HF SegFormer encoder weights (MiT) → local `mit.py`
This creates a checkpoint `checkpoints/mit_b2_from_hf.pth` with weights for our MiT backbone.

```bash
python scripts/load_from_hf.py --hf-id nvidia/segformer-b2-finetuned-ade-512-512 --variant mit_b2 --out checkpoints/mit_b2_from_hf.pth
```

## Use converted weights
You can **initialize the backbone** with converted weights via `--resume` (full model) or by loading just the backbone:

- Easiest: resume a full-model checkpoint that already contains backbone weights.
- Or modify `train.py` to load backbone weights only (see comment in file). A basic example:

```bash
# quick demo: run one forward to initialize shapes, then load backbone
python - <<'PY'
import torch
from backbones import create_pyramid_backbone
from segformer import SegFormer
bk, ch, _ = create_pyramid_backbone("mit_b2")
model = SegFormer(bk, ch, num_classes=19)
sd = torch.load("checkpoints/mit_b2_from_hf.pth", map_location="cpu")
model.backbone.load_state_dict(sd, strict=True)
print("Loaded MiT weights from HF into local backbone.")
PY
```

## Inference
```bash
python infer.py --weights runs/exp_mit/best.pth --images /path/to/test_images --backbone mit_b2 --num-classes 19 --out out_vis
```

Notes:
- HF conversion loads the **encoder** (MiT) weights only; the decoder remains randomly initialized unless you've trained it.
- Strides are [4, 8, 16, 32]; channels follow variant embed dims.
