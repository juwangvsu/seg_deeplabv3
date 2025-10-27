
# Radar-based Semantic Segmentation (SegFormer backbone)

This mini-repo contains:
- `radar_dataset.py` — dataset for float32 range–angle `.npy` plus PNG masks (and optional RGB overlays).
- `model_design.py` — SegFormer wrapper configured for **single-channel** radar input.
- `utils_io.py` — checkpointing and prediction image saving (raw/colored/overlay).
- `train_eval_infer.py` — CLI for **train / eval / infer** modes with `--save-every`, `--ckpt-dir`, `--load`, and `--save-output`.

## Expected data layout
```
data_dir/
  angle_range_numpy/*.npy      # float32 HxW range–angle maps
  masks/*.png                  # uint8 class IDs, same stem as npy
  images/*.png|*.jpg           # optional RGB (for overlays)
```

---------------10/27/25 rad_segformer --------------------
@gpu1:
Train:
python3 train_eval_infer.py   --data-dir ../data/radar_scorp/   --num-classes 19   --mode train   --epochs 50   --batch-size 8   --image-size 512   --save-every 2   --ckpt-dir checkpoints/radar_b2   --out-dir outputs/radar_b2

Eval (and save outputs):
python3 train_eval_infer.py   --data-dir ../data/radar_scorp   --num-classes 19   --mode eval   --load best.pt   --ckpt-dir checkpoints/radar_b2   --save-output   --out-dir outputs/radar_b2

Infer one file:
python3 train_eval_infer.py   --data-dir ../data/radar_scorp   --num-classes 19   --mode infer   --load best.pt   --ckpt-dir checkpoints/radar_b2   --input-file /path/to/data/angle_range_numpy/000123.npy   --out-dir outputs/radar_b2


Epoch 10/50 | Val mIoU=0.2217 Acc=0.8220
Epoch 11/50 | Val mIoU=0.2212 Acc=0.8207
Epoch 12/50 | Val mIoU=0.2048 Acc=0.8245
Epoch 13/50 | Val mIoU=0.2349 Acc=0.8400
Epoch 14/50 | Val mIoU=0.2263 Acc=0.8362
Epoch 15/50 | Val mIoU=0.2429 Acc=0.8501
Epoch 16/50 | Val mIoU=0.2501 Acc=0.8501
Epoch 17/50 | Val mIoU=0.2492 Acc=0.8568
Epoch 18/50 | Val mIoU=0.2420 Acc=0.8480
Epoch 19/50 | Val mIoU=0.2461 Acc=0.8457
Epoch 20/50 | Val mIoU=0.2706 Acc=0.8744
Epoch 21/50 | Val mIoU=0.2728 Acc=0.8709
Epoch 22/50 | Val mIoU=0.2696 Acc=0.8641
Epoch 23/50 | Val mIoU=0.2815 Acc=0.8761
Epoch 24/50 | Val mIoU=0.2835 Acc=0.8839
Epoch 25/50 | Val mIoU=0.2833 Acc=0.8786
Epoch 26/50 | Val mIoU=0.2787 Acc=0.8839
Epoch 27/50 | Val mIoU=0.2894 Acc=0.8876
Epoch 48/50 | Val mIoU=0.3206 Acc=0.9147
Epoch 49/50 | Val mIoU=0.3220 Acc=0.9138
Epoch 50/50 | Val mIoU=0.3177 Acc=0.9122
Notes:
- The model is configured for **1-channel** inputs. To leverage RGB-pretrained weights, pass a `pretrained_name` into `RadarSegFormer` and it will average the first conv weights to 1ch.
- Ignore index is **255** for masks.
- Predictions go to `masks/`, `colored_masks/`, and `overlay/` under the chosen output directory.
