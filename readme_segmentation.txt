https://chatgpt.com/share/e/68e53f50-2e9c-800c-9714-8bbd42ab2cd3

----------------------9/27/25 sticky--------------
alien3:ws3
	student@alien3:~/Documents/parking-seg-deeplabv3
docker:
	jwang3vsu/parking-seg:cuda12.1
	container:
		deeplabseg
data:alien3
	/media/student/datar/radarstuff/20190813_scorp_dataset
	~/Documents/datasets/radar_cam
outputs:
	segformer/radar_scorp : segformer prediction, as gt
	tmp3		:       deeplabv3 retrained model prediction on radarstuff
	tmp4		:       deeplabv3 retrained model prediction on degraded radarstuff

diff_:
	compare deeplabv3 prediction vs gt(segformer result)
	python3 diff_images.py --dir1 outputs/tmp3/color_mask/ --dir2 outputs/segformer/radar_scorp/colored_masks/ --outdir tmp/out/

versions:
	segformer_mitlocal:	local impl mit backbone + local simple head
	pvtv2_seg_city2:	local pvtv2 backbone impl + local head
	segformer_pvtv2_min:	stock timm pvtv2 backbone + local head
	segformer_hf_test:	hf segformer with stock nvidia weights for ade or cityscapes

eval:
	segformer_hf_test:b0:b2:b3:
		Pixel Accuracy: 0.78%
		Mean IoU:       0.63%
	label id should be fixed in both train and infer. need to filter out
	some labelid in original dataset to keep 19, see segformer_pvtv2_min
	train.py and data_city.py

current:
	10/14: compare training result on cityscapes: stock pvtv2 vs local...
	   python3 pvtv2_seg_city2/inference.py --images-dir data/cityscape/leftImg8bit/val/ --load runs/exp/best.pth   --encoder b2 --num-classes 19 --size 512 896 --out-dir runs/exp2_infer --overlay 
	   eog runs/exp2_infer/color_mask/frankfurt_000000_009688_leftImg8bit.png

	   python3 segformer_pvtv2_min/train.py --config segformer_pvtv2_min/config_city.yml --encoder pvt_v2_b2 [--load outputs_segformref_city/segformer_local.pth] 


	10/1/25 segformer process radarseg image, train deeplabv3		and compare result

-----------4/3/26 retest rad_segformer ---------------------------

eval/infer outputs:
	cp to /media/student/datar/radarstuff/20190813_scorp_dataset/outputs/
	
view outputs from i9:
	student@i9ub22:~/gpudata/jwang/datasets$ eog datar/radarstuff/20190813_scorp_dataset/outputs/radar_b2_retest/infer_outputs/overlay/000123.png 

-----------4/3,6/26 recap generate gt dataset ---------------------------
deeplabseg@gpu1:

(gt)radar_scorp:
	python3 test_seg4.py --data_dir data/radar_scorp --out_dir outputs/segformer/radar_scorp2

(gt)k-radar:
	data prep:
		python3 crop_left.py /data/jwang/datasets/k-radar/18/
		this generate cam-front-left/
	python3 test_seg4.py --data_dir /media/student/datar/k-radar/18/cam-front-left  --out_dir /media/student/datar/k-radar/18/segformer/k-radar-18
		colored_masks, masks, overlay

test_seg4.py use stock segformer to generate segmentation mask

generated:
	radar_scorp/masks/ (9/27/25 note)
	
-----------3/30/26 retest radseg ---------------------------
deeplabseg@gpu1:

data access from alien3 via sshfs:
	alien3:~/gpudata
	sshfs gputest@gpuhead:/data gpudata
	eog ~/gpudata/jwang/Documents/seg_deeplabv3/radseg/checkpoints/preds_eval/000372_overlay.png
	
-----------10/24/25 radseg ---------------------------
use radar only for segmentation
Epoch 001: loss=1.4223 acc=0.7474 miou=0.1052
Epoch 002: loss=0.7739 acc=0.7595 miou=0.1222
Epoch 003: loss=0.6644 acc=0.7307 miou=0.1526
Epoch 004: loss=0.6246 acc=0.7127 miou=0.1451

see readme_radseg.txt
result is supprising good

radar data visualize:
 	python3 convert_angle_npy_to_png.py -i data/radar_scorp/angle_range_numpy/000321.npy -o 000321.png --autoscale
	Convert an angle_range NumPy file (.npy) to a pseudo-colored PNG.

-----------10/15/35 /workspace/segformer_hf_test#----------
python3 infer_segformer.py   --model-id nvidia/segformer-b0-finetuned-cityscapes-512-1024   --input ../data/cityscape/leftImg8bit/val/lindau/lindau_000019_000019_leftImg8bit.png --out-dir out_city
	stock nvidia pretrained weights.

-------10/13/25 segformer_mitlocal --------------

alien3:ws3

img_size 512x896, dataset: if cityscape, all imgs under train/ and val. train loop split randomly to train and val

train:
   pvtv2:
	segformer_mitlocal# 
	python3 train.py --data-root ../data/apgdata --backbone pvt_v2_b2 --num-classes 19 --epochs 40 --batch-size 6 --img_size 768 768 --out runs/exp_mit_pvt2
	Epoch 40/40 | loss 0.0570 | mIoU 0.3460

	python3 train.py --data-root ../data/cityscape --backbone pvt_v2_b2 --num-classes 19 --epochs 40 --batch-size 6 --out runs/exp_mit_pvt2_city --dstype cityscapes  
	Epoch 40/40 | loss 0.1285 | mIoU 0.4110

   mitbone:
	python3 train.py --data-root ../data/apgdata --backbone mit_b2 --num-classes 19 --epochs 40 --batch-size 6 --out runs/exp_mit_mitbone

	?python3 train.py --data-root ../data/cityscape --backbone mit_b2 --num-classes 19 --epochs 40 --batch-size 6 --img-size 768 --out runs/exp_mit_mitbone_city --dstype cityscapes [--boverride checkpoints/mit_b2_from_hf.pth] 
	Epoch 40/40 | loss 0.1928 | mIoU 0.4665

infer:
   pvtv2:
	python infer.py --weights runs/exp_mit_pvt2/best.pth --images ../data/apgdata/images --backbone pvt_v2_b2 --num-classes 19 --out runs/exp_mitpvt_infer 
	python3 infer.py --weights runs/exp_mit_pvt2_city/best.pth --images ../data/cityscape/leftImg8bit/val/frankfurt --backbone pvt_v2_b2 --num-classes 19 --out runs/exp_mitpvt_infer_city
 
   mitbone:
	python3 infer.py --weights runs/exp_mit_mitbone/best.pth --images ../data/apgdata/images --backbone mit_b2 --num-classes 19 --out runs/exp_mitbackone_infer

-------10/11/25 pvtv2_seg_city2/ ws3 gpu1--------------
/home/sysinit/Documents/datasets/datar
/home/sysinit/Documents/datasets/datarad
/home/sysinit/Documents/seg_deeplabv3
	pvtv2_seg_city2
	dataset: cityscapes only, train for train, val for val
docker:(ws3)
	docker run --gpus all -t -d --restart always --privileged --shm-size=1g  -v /home/sysinit/Documents/datasets/datar:/media/student/datar -v /home/sysinit/Documents/datasets/datarad:/media/student/datarad -v $PWD/samples:/workspace/samples   -v $PWD/outputs:/workspace/outputs -v $PWD:/workspace  -e DISPLAY=${DISPLAY} -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v ${XAUTHORITY:-$HOME/.Xauthority}:/root/.Xauthority --name deeplabseg jwang3vsu/parking-seg:cuda12.1  bash

docker:(gpu1)
	docker run --gpus all -t -d --shm-size=1g  -v /data/jwang/datasets/datar:/media/student/datar -v /data/jwang/datasets/datarad:/media/student/datarad -v $PWD/samples:/workspace/samples   -v $PWD/outputs:/workspace/outputs -v $PWD:/workspace --name deeplabseg jwang3vsu/parking-seg:cuda12.1  bash
		/workspace/data/cityscape# ln -sn /media/student/datarad/city/leftImg8bit/
		/workspace/data/cityscape# ln -sn /media/student/datarad/city_gtfine/gtFine/

-------10/11/25 pvtv2_seg_city2/ alien3--------------
train:
	python3 pvtv2_seg_city2/train.py --dataset cityscapes --data-root data/cityscape   --encoder b2 --num-classes 19 --size 512 1024 --epochs 100 --batch-size 8 --load runs/exp/best.pth [--out-dir runs/exp]
	80 sec/epoch
	train_loss=0.1339  val_acc=0.9285  val_mIoU=0.5470
	
infer:
	python3 pvtv2_seg_city2/inference.py --images-dir data/cityscape/leftImg8bit/val/ --load runs/exp/best.pth   --encoder b2 --num-classes 19 --size 512 896 --out-dir runs/exp2_infer --overlay --showmodel

runs/exp3_infer:
	inferience overlay image incorrect

backbone output shape, decoded final output shape:
	torch.Size([4, 64, 128, 224]) torch.Size([4, 128, 64, 112]) torch.Size([4, 320, 32, 56]) torch.Size([4, 512, 16, 28]) torch.Size([4, 19, 512, 896])
tbd:
	check model structure btw this model and official backbone.
	model print show when they are created? not necessary follow the computing flow in forward.	

-------10/9/25  segformer_pvt_min train radar_scorp image only----------

segformer_pvtv2_min/
	stock backbone (timm), local decode head
	backbone = timm.create_mode()
	decode_head = SegFormerHead
	
dataset:
	data.py for regular, data_city.py for cityscape type, select via yaml
deep:
	python3 segformer_pvtv2_min/train.py --config segformer_pvtv2_min/config.yml --encoder pvt_v2_b2 --load outputs_segformref/segformer_local.pth 
		data/radar_scorp
		outputs_segformeref/segformer_radarscorp_imgs.pth
	python3 segformer_pvtv2_min/train.py --config segformer_pvtv2_min/config_city.yml --encoder pvt_v2_b2 [--load outputs_segformref_city/segformer_local.pth] 
		data/cityscape
		outputs_segformref_city/*.pth

infer:
	python3 segformer_pvtv2_min/infer.py --images data/radar_scorp/images \
  --checkpoint outputs_segformref/segformer_radarscorp_imgs.pth \
  --out_dir outputs_infer_radarscorp2 --encoder pvt_v2_b2 --input_size 768 [--showmodel]


--------10/5/25 radcam --------------------
root@96c5e2aa4727:/workspace# python3 -m radcam.train --config configs/radarcam.yaml [--eval]

status:
	loss nan
	fixed, to double check what cause nan
	radcam_model.pt
	load pt issue, might be model changed?
	save orig incorrect. radcam/eval.py:83

decode head might need relu and batchnorm:
    (1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): ReLU()
    (4): Conv2d(256, 21, kernel_size=(1, 1), stride=(1, 1))


---------10/1/25 segformer/city train deeplabv3 radarcam dataset' camera -------
	(-1) dataset: cp subfolder camxxx to images
	(0) cd /media/student/datar/radarstuff/20190813_scorp_dataset/20190813_icmim_dataset/2019-08-13-14-04-36/images
		mogrify -format png *.jpg
	(1) sgf1 : generate gt using segformer
	    python3 verify_label.py --masks outputs/segformer/radar_scorp2/masks/
		~/Documents/parking-seg-deeplabv3$ ./rename_mask2.sh 

	    cp -r outputs/segformer/radar_scorp2/masks/ /media/student/datar/radarstuff/20190813_scorp_dataset/20190813_icmim_dataset/2019-08-13-14-04-36/

	
	(2) train:
		python3 train.py --config configs/radarcam.yaml

	(3) predict:
	python3 predict_deeplabv3.py --input_dir /media/student/datar/radarstuff/20190813_scorp_dataset/20190813_icmim_dataset/2019-08-13-14-04-36/ --load_ckpt --ckptfn outputs/best.ckpt --output_dir outputs/tmp3 [--filename 000167.pmg]
		single filename mode. also in test_seg4.py  


issue:
	color palette diff between predict_deeplabv3.py and test_seg4.py
	fixed:  unified to same color palette

status:
	retrained result decent. next to 
	(1) degrade image and eval prediction accuracy
	(2) try to train and predict using radar image
... 

---------9/30/25 segformer/city train deeplabv3 apgdata -------
test_seg4.py:
	pretrained weights city scrape
segformer pretrained model (cityscrape) generate mask as gt to train
deeplabv3 model:
   58  python3 test_seg4.py --data_dir data/apgdata --out_dir outputs/segformer_city
   59  python3 verify_label.py --masks outputs/segformer_city/masks/
 2032  eog outputs/segformer_city/overlay/000361_overlay.png 
 2033  sudo rm data/apgdata/masks/* -f
 2034  sudo cp  outputs/segformer_city/masks/* data/apgdata/masks/

   60  cd data/apgdata/masks/
   61  ../../../rename_mask.sh 
   64  python3 train.py --config configs/apgdata.yaml
   71  python3 predict_deeplabv3.py --load_ckpt --ckptfn outputs/epoch_010.ckpt --output_dir outputs/tmp2

--------------9/27/25 test stock deeplabv3 ----------------------
use pretrained model:
@alien3:host
@deeplabseg:
	python3 predict_deeplabv3.py --load_ckpt --ckptfn model_deeplabv3_pretrained.pt --output_dir outputs/tmp2
		load stock or local deeplabv3 parameters, predict result saved
		input dir data/apgdata/images
 	python3 verify_label.py --masks outputs/tmp2/masks/
		good.
	chown -R stuent data/apgdata
	~/Documents/parking-seg-deeplabv3/data/apgdata/masks$ ../../../rename_mask.sh 
		rename masks filename
     	python3 train.py --config configs/parkinglot.yaml
	python3 train.py --config configs/apgdata.yaml

	python3 verify_label.py --masks data/apgdata/masks

segformer:
       (sgf1) python3 test_seg4.py --data_dir  /media/student/datar/radarstuff/20190813_scorp_dataset/20190813_icmim_dataset/2019-08-13-14-04-36 --out_dir outputs/segformer/radar_scorp2 --device cuda

        python3 test_seg4.py --data_dir data/apgdata --out_dir outputs/segformer
		assume images under data_dir
        	use nvidia/segformer-b0-finetuned-ade-512-512 not deeplabv3.
		result saved in 
			data_dir: masks, overlay, colored_masks
status:
	apgdata training run works now. 
	python3 verify_label.py --masks data/apgdata/masks
		some not valid, should not happen since using stock weights

@@	 python3 visualize.py --images data/apgdata/images   --masks  data/apgdata/masks --output_dir outputs/vis_train4   --legend --alpha 0.5
	eog outputs/vis_train4/000464_overlay.png 
	!!!predict_deeplabv3.py color_mask only show person

train apgdata issue:
	    scaler.scale(loss).backward()
  File "/usr/local/lib/python3.10/dist-packages/torch/amp/grad_scaler.py", line 214, in scale
    return outputs * self._scale.to(device=outputs.device, non_blocking=True)
torch.AcceleratorError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

cause:
	That CUDA assert almost always means a label in your mask is outside [0..num_classes-1] and not equal to ignore_index. For your config (num_classes: 5, ignore_index: 255), any pixel not in {0,1,2,3,4,255} will crash CrossEntropyLoss on GPU.

	https://chatgpt.com/share/e/68da0dc3-527c-800c-a6f3-b1a6552f221d

-------------9/27/25 loading stock param to local model -----
issue:
	Missing key(s) in state_dict: "aux_classifier.0.weight", "aux_classifier.1.weight", "aux_classifier.1.bias", "aux_classifier.1.running_mean", "aux_classifier.1.running_var", "aux_classifier.4.weight", "aux_classifier.4.bias".

fix:
	deeplabv3 has an aux_classifier head. when loading an empty model, aux 
	is not loaded. when loading a pretrained model, it usually have aux head.
	relevant aug when loading: model.py
	model = torchvision.models.segmentation.deeplabv3_resnet50(
        aux_loss=True
    	)
	this will create a model with aux_classifier

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
	cd ~/Documents/parking-seg-deeplabv3
	docker run --gpus all -t -d --shm-size=1g  -v /media/student/datar:/media/student/datar -v /media/student/datarad:/media/student/datarad -v $PWD/samples:/workspace/samples   -v $PWD/outputs:/workspace/outputs -v $PWD:/workspace --name deeplabseg jwang3vsu/parking-seg:cuda12.1  bash
		pip install transformers==4.56.2
	
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
