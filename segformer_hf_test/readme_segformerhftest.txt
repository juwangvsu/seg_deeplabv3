status:
	random init finetuing... ./segformer_city_ft_random/
	adding dpp version
	double check infer_... acc and miou calculation
		seems not righ
		the unique values in mask contain >19, not right....
python3 check_labels.py --mask_dir ../data/cityscape/gtFine/val/ --recursive
Unique label IDs across all masks:
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]

Total unique IDs: 33

In the Cityscapes dataset, the label IDs 32 and 33 represent "void" classes that are not included in the standard 30-class definition. The dataset creators added them to represent special cases during the annotation process. 
The Cityscapes dataset distinguishes between the raw annotation IDs (which go up to 33) and the 19 standard classes used for model training and evaluation. The IDs 32 and 33 are part of a special "void" group that also includes other ignored classes like ego vehicle and rectification border

-------------------------------------------------------
https://huggingface.co/nvidia/mit-b4
https://huggingface.co/models?other=segformer

segformer hf nvidia pretrained.
 segformer official mit_b2

# ADE20K pretrained model, single image
python infer_segformer2.py \
  --model-id nvidia/segformer-b2-finetuned-ade-512-512 \
  --input path/to/image.jpg \
  --out-dir out_ade

# Batch a folder; keep original resolution
python infer_segformer2.py \
  --model-id nvidia/segformer-b5-finetuned-ade-640-640 \
  --input ./images \
  --out-dir out_batch

python infer_segformer2.py \
  --model-id /path/to/local-segformer \
  --input ./cityscapes_samples \
  --out-dir out_city

python3 infer_segformer2.py   --model-id nvidia/segformer-b0-finetuned-cityscapes-512-1024   --input ../data/cityscape/leftImg8bit/val --out-dir out_city --save --showmodel --mask-dir ../data/cityscape/gtFine/val

python3 infer_segformer2.py   --model-id nvidia/segformer-b0-finetuned-cityscapes-512-1024   --input-dir ../data/cityscape/leftImg8bit/val/frankfurt/frankfurt_000000_012868_leftImg8bit.png --out-dir out_city [--showmodel] [--save] [--mask-dir ]
	nvidia/segformer-b2-finetuned-cityscapes-1024-1024


https://huggingface.co/models?other=segformer

nvidia/segformer-b0-finetuned-cityscapes-1024-1024 
nvidia/segformer-b2-finetuned-cityscapes-1024-1024 
nvidia/segformer-b3-finetuned-cityscapes-1024-1024 
nvidia/segformer-b4-finetuned-cityscapes-1024-1024 
nvidia/segformer-b5-finetuned-cityscapes-1024-1024

finetune:
	python3 train_finetune_segformer2.py \
  --model-id nvidia/segformer-b2-finetuned-cityscapes-1024-1024 \
  --train-input-dir ../data/cityscape/leftImg8bit/train \
  --train-mask-dir  ../data/cityscape/gtFine/train \
  --val-input-dir   ../data/cityscape/leftImg8bit/val \
  --val-mask-dir    ../data/cityscape/gtFine/val \
  --output-dir      ./segformer_city_ft \
 [--rand-backbone]  --batch-size 8 --lr 6e-5 --epochs 30 --img-height 512 --img-width 1024 --fp16

---------------------------------------------
random backbone: ws3

python3 train_finetune_segformer2.py   --model-id nvidia/segformer-b2-finetuned-cityscapes-1024-1024   --train-input-dir ../data/cityscape/leftImg8bit/train   --train-mask-dir  ../data/cityscape/gtFine/train   --val-input-dir   ../data/cityscape/leftImg8bit/val   --val-mask-dir    ../data/cityscape/gtFine/val   --output-dir      ./segformer_city_ft_random   --batch-size 8 --lr 6e-5 --epochs 30 --img-height 512 --img-width 1024 --fp16 --rand-backbone --resume-from ./segformer_city_ft_random/best.pth 
	[val] epoch 30: loss=0.4874, acc=85.82%, mIoU=31.31%
        Epoch 100/100 | step 360/372 | loss 0.1632
[val] epoch 150: loss=0.5975, acc=88.52%, mIoU=37.41%
[val] epoch 200: loss=0.6355, acc=88.52%, mIoU=37.09%

-----------------------------------------------------
random head: gpu1

python3 train_finetune_segformer2.py   --model-id nvidia/segformer-b2-finetuned-cityscapes-1024-1024   --train-input-dir ../data/cityscape/leftImg8bit/train   --train-mask-dir  ../data/cityscape/gtFine/train   --val-input-dir   ../data/cityscape/leftImg8bit/val   --val-mask-dir    ../data/cityscape/gtFine/val   --output-dir      ./segformer_city_ft_randomhead   --batch-size 8 --lr 6e-5 --epochs 200 --img-height 512 --img-width 1024 --fp16 --rand-decode-head --resume-from ./segformer_city_ft_randomhead/best.pth 
[val] epoch 1: loss=0.2150, acc=93.81%, mIoU=61.98%
        epoch 1: loss 3.6 -> 0.19
[val] epoch 15: loss=0.1493, acc=95.79%, mIoU=75.70%
[val] epoch 30: loss=0.1547, acc=95.87%, mIoU=76.33%

---------------------------------------------------------------------
infer:
	python3 infer_segformer2.py  --model-id nvidia/segformer-b2-finetuned-cityscapes-1024-1024 --load segformer_city_ft/best.pth   --input-dir ../data/cityscape/leftImg8bit/val --out-dir out_city_b2_ft --mask-dir ../data/cityscape/gtFine/val

       ws3:
        python3 infer_segformer2.py  --model-id nvidia/segformer-b2-finetuned-cityscapes-1024-1024 --load segformer_city_ft_random/best.pth   --input-dir ../data/cityscape/leftImg8bit/val --out-dir out_city_b2_ft_random --mask-dir ../data/cityscape/gtFine/val

        gpu1:
	python3 infer_segformer2.py  --model-id nvidia/segformer-b2-finetuned-cityscapes-1024-1024 --load segformer_city_ft_randomhead/epoch_001.pth   --input-dir ../data/cityscape_quick/leftImg8bit/val --out-dir out_city_b2_ft_randomhead --mask-dir ../data/cityscape_quick/gtFine/val
		epoch 30 good quality
		epoch 2 still decent

10/20/25
	gt label mapping fixed
	miou calculation fixed, none present class excluded in compute_metrics
========== Cityscapes Evaluation ==========
random:(backbone)

epoch_001, data/cityscape/leftImg8bit/val
Pixel Accuracy: 52.04% 0.5203552618882911
Mean IoU:       8.33% 0.08325688679922241

epoch_001, data/cityscape/leftImg8bit/train
Pixel Accuracy: 51.74% 0.5173961929199637
Mean IoU:       8.68% 0.0868003520369801

epoch_001, data/cityscape_quick/leftImg8bit/val
Pixel Accuracy: 50.31% 0.5031147261973504
Mean IoU:       7.89% 0.07888204649779312

epoch_030, data/cityscape/leftImg8bit/val
Pixel Accuracy: 83.08% 0.830773510172923
Mean IoU:       28.46% 0.2845551495153419

epoch_030, data/cityscape/leftImg8bit/train
Pixel Accuracy: 85.76% 0.8575685676246924
Mean IoU:       35.55% 0.3555067316471587

epoch_030, data/cityscape_quick/leftImg8bit/val
Pixel Accuracy: 80.88% 0.808755995662897
Mean IoU:       24.30% 0.24303238778258124

epoch_200, data/cityscape/leftImg8bit/train
Pixel Accuracy: 89.64% 0.8964215264304525
Mean IoU:       45.54% 0.45538206000051495

epoch_200, data/cityscape/leftImg8bit/val
iPixel Accuracy: 86.61% 0.8661042067604375
Mean IoU:       33.81% 0.33805307502297877

randomhead
best
Pixel Accuracy: 92.66% 0.9265750563369761
Mean IoU:       45.95% 0.45945491797928883

epoch_001, data/cityscape/leftImg8bit/val
Pixel Accuracy: 91.77% 0.9177280077719349
Mean IoU:       54.20% 0.5419769881072555

epoch_001, data/cityscape/leftImg8bit/train
...

epoch_001, data/cityscape_quick/leftImg8bit/val
Pixel Accuracy: 90.87% 0.9086916349097909
Mean IoU:       39.32% 0.39316229213734416

epoch_031, data/cityscape/leftImg8bit/val
Pixel Accuracy: 94.00% 0.9400259813082132
Mean IoU:       66.63% 0.6663183285081321

epoch_031, data/cityscape/leftImg8bit/train
Pixel Accuracy: 95.88% 0.9587838864632993
Mean IoU:       76.57% 0.7656998569397269

epoch_031, data/cityscape_quick/leftImg8bit/val
Pixel Accuracy: 92.58% 0.9258156881296874
Mean IoU:       46.03% 0.4603321652496467

b2 model:
b2 model, data/cityscape/leftImg8bit/val
Pixel Accuracy: 93.28% 0.9327920857220579
Mean IoU:       65.33% 0.6532851414570748

b2 model, data/cityscape/leftImg8bit/train
Pixel Accuracy: 94.79% 0.947860495975673
Mean IoU:       73.15% 0.7314871720855852

b2 model, data/cityscape_quick/leftImg8bit/val
Pixel Accuracy: 90.30% 0.9030087884659528
Mean IoU:       42.93% 0.4292624184714141

-----------------------------------------
        eog:
        segformer_hf_test$ eog out_city_b2_ft/overlay/frankfurt/frankfurt_000000_000294_leftImg8bit.png

