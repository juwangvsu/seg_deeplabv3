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

python3 infer_segformer2.py   --model-id nvidia/segformer-b0-finetuned-cityscapes-512-1024   --input ../data/cityscape/leftImg8bit/val/frankfurt/frankfurt_000000_012868_leftImg8bit.png --out-dir out_city 
	nvidia/segformer-b2-finetuned-cityscapes-1024-1024

https://huggingface.co/models?other=segformer

nvidia/segformer-b0-finetuned-cityscapes-1024-1024 
nvidia/segformer-b2-finetuned-cityscapes-1024-1024 
nvidia/segformer-b3-finetuned-cityscapes-1024-1024 
nvidia/segformer-b4-finetuned-cityscapes-1024-1024 
nvidia/segformer-b5-finetuned-cityscapes-1024-1024
