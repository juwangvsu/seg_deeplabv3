status:
	random init finetuing... ./segformer_city_ft_random/

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

python3 train_finetune_segformer2.py   --model-id nvidia/segformer-b2-finetuned-cityscapes-1024-1024   --train-input-dir ../data/cityscape/leftImg8bit/train   --train-mask-dir  ../data/cityscape/gtFine/train   --val-input-dir   ../data/cityscape/leftImg8bit/val   --val-mask-dir    ../data/cityscape/gtFine/val   --output-dir      ./segformer_city_ft_random   --batch-size 8 --lr 6e-5 --epochs 30 --img-height 512 --img-width 1024 --fp16 --rand-backbone --resume-from ./segformer_city_ft_random/best.pth 
	[val] epoch 30: loss=0.4874, acc=85.82%, mIoU=31.31%

python3 infer_segformer2.py  --model-id nvidia/segformer-b2-finetuned-cityscapes-1024-1024 --load segformer_city_ft/best.pth   --input-dir ../data/cityscape/leftImg8bit/val --out-dir out_city_b2_ft --mask-dir ../data/cityscape/gtFine/val
