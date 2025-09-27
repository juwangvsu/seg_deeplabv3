
----------------------9/27/25 sticky--------------
alien3:
	student@alien3:~/Documents/parking-seg-deeplabv3
docker:
	jwang3vsu/parking-seg:cuda12.1
	container:
		deeplabseg
---------------------------------------------------------------------------------------
# Parking Lot Segmentation (DeepLabv3)

Train and evaluate **DeepLabv3 (ResNet-50)** for semantic segmentation on a parking-lot dataset
(classes: pavement, person, car, tree; plus background).

- Host OS: **Ubuntu 22.04**
- GPU: **NVIDIA RTX 4090**
- Container: CUDA 12.1 runtime + PyTorch 2.x (CUDA 12.1 wheels)
- Image resolution: 1232 × 1028 (we train with random scales/crops around this size)

## Dataset layout

Place your dataset under `data/` (you can mount it into the container). Expected structure:

```
data/
  train/
    images/   # *.jpg or *.png
    masks/    # *.png (single-channel, integer class IDs)
  val/
    images/
    masks/
  test/
    images/
    masks/    # optional; if omitted, eval runs only on val
```

- File names in `images/` and `masks/` must match (e.g., `0001.png` ↔ `0001.png`).
- Mask pixel values are class IDs in **[0..4]** with `255` as ignore label (optional).

### Class map

```
0: background
1: pavement
2: person
3: car
4: tree
```

If you already encode background as 0 and the four classes as 1–4, you’re good. Otherwise,
adapt `configs/parkinglot.yaml` (the `num_classes` and `class_names` fields) and/or preprocess masks.

## Quick start

### 1) Build the Docker image
```bash
docker build -t parking-seg:cuda12.1 -f Dockerfile .
```

1.5)
	docker run --gpus all -t -d   -v $PWD/samples:/workspace/samples   -v $PWD/outputs:/workspace/outputs -v $PWD:/workspace --name deeplabseg parking-seg:cuda12.1  bash

### 2) Train
Mount your dataset and an output directory:
```bash
docker run --gpus all --rm -it   -v $PWD/data:/workspace/data   -v $PWD/outputs:/workspace/outputs   parking-seg:cuda12.1   python train.py --config configs/parkinglot.yaml
```

### 3) Evaluate
```bash
docker run --gpus all --rm -it   -v $PWD/data:/workspace/data   -v $PWD/outputs:/workspace/outputs   parking-seg:cuda12.1   python eval.py --config configs/parkinglot.yaml --checkpoint outputs/latest.ckpt
```

### 4) Predict on a folder of images (optional)
```bash
docker run --gpus all --rm -it   -v $PWD/samples:/workspace/samples   -v $PWD/outputs:/workspace/outputs   parking-seg:cuda12.1   python scripts/predict_folder.py --checkpoint outputs/latest.ckpt --input_dir samples --output_dir outputs/preds
```
visualize:
python scripts/visualize.py \
  --images data/train/images \
  --masks  data/train/masks \
  --output_dir outputs/vis_train \
  --legend --alpha 0.5

## Config knobs (see `configs/parkinglot.yaml`)
- `num_classes`: 5 (background + 4 classes)
- `input_size`: training crop size (default 768)
- `batch_size`: default 4 (fits 24GB+; adjust for your GPU/VRAM)
- `lr`: default 6e-4 (AdamW)
- `max_epochs`: default 60
- `ignore_index`: 255

## Notes
- Uses mixed precision (AMP) by default.
- SyncBN disabled by default for single-GPU; can be enabled for multi-GPU (DDP not wired here to keep it simple).
- If masks look ragged on tree branches, consider increasing `input_size` or adding a boundary loss (left as an exercise).

---

**Enjoy!**
