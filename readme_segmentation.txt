
----------------------9/27/25 sticky--------------
alien3:
	student@alien3:~/Documents/parking-seg-deeplabv3
docker:
	jwang3vsu/parking-seg:cuda12.1
	container:
		deeplabseg

--------------9/27/25 test stock deeplabv3 ----------------------
use pretrained model:

python3 test_seg3.py --input 000010.png   --output result_overlay.png   --colored result_colored_mask.png

python3 test_seg4.py --data_dir data/kitti_object_100

-------------9/27/25 loading stock param to local model -----
issue:
	Missing key(s) in state_dict: "aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias", "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.4.weight", "aux_classifier.4.bias".

fix:
	deeplabv3 has an aux_classifier head. when loading an empty model, aux 
	is not loaded. when loading a pretrained model, it usually have aux head.
	relevant aug when loading: model = torchvision.models.segmentation.deeplabv3_resnet50(
        aux_loss=False
    )

---------------------deeplabv3 local train -----------------------------------

Train and evaluate **DeepLabv3 (ResNet-50)** for semantic segmentation on a parking-lot dataset
(classes: pavement, person, car, tree; plus background).

- Image resolution: 1232 × 1028 (we train with random scales/crops around this size)

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
  File names in `images/` and `masks/` must match
- Mask pixel values are class IDs in **[0..4]** with `255` as ignore label (optional).

Class map:
	0: background
	1: pavement
	2: person
	3: car
	4: tree

docker build -t parking-seg:cuda12.1 -f Dockerfile .

docker:
	docker run --gpus all -t -d   -v $PWD/samples:/workspace/samples   -v $PWD/outputs:/workspace/outputs -v $PWD:/workspace --name deeplabseg parking-seg:cuda12.1  bash

Train:
	python train.py --config configs/parkinglot.yaml

Evaluate:
	python eval.py --config configs/parkinglot.yaml --checkpoint outputs/latest.ckpt

Predict:
	python3 predict_folder.py --checkpoint outputs/latest.ckpt --input_dir data/apgdata/images --output_dir outputs/apgdata/preds

visualize:
	python3 visualize.py --images data/apgdata/images   --masks  outputs/apgdata/preds --output_dir outputs/vis_train2   --legend --alpha 0.5 --maskpref mask

## Config knobs (see `configs/parkinglot.yaml`)
- `num_classes`: 5 (background + 4 classes)
- `input_size`: training crop size (default 768)
- `batch_size`: default 4 (fits 24GB+; adjust for your GPU/VRAM)
- `lr`: default 6e-4 (AdamW)
- `max_epochs`: default 60
