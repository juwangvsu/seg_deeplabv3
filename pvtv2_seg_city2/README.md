# PVTv2 + SegHeadLite (Minimal Segmentation Starter)

Includes Cityscapes-ready training, config YAMLs, and batch inference.

## Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Train on folder dataset
```bash
python train.py --config configs/train_folder_example.yaml
# or CLI:
python train.py --data-root /data/seg --dataset folder --encoder b2 --num-classes 19 --size 512 896
```

## Train on Cityscapes
```bash
python train.py --config configs/train_cityscapes.yaml
# or CLI:
python train.py --dataset cityscapes --data-root /path/to/cityscapes   --cs-mode auto --encoder b2 --num-classes 19 --size 512 1024
```

## Inference
```bash
python inference.py --config configs/infer_example.yaml
# or CLI:
python inference.py --images-dir /data/seg/val/images --load runs/exp1/best.pth   --encoder b2 --num-classes 19 --size 512 896 --overlay
```
