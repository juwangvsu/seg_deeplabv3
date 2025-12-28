backbone:
	locally defined segformer backbone

train input: radar range_angle npy
training gt: semantic segmentation mask (human labeled or predicted by segformer) 

gpu1:

deeplabseg2
root@2d5dc237209e:/workspace/radseg

train:
	root@2d5dc237209e:/workspace/radseg# 
	python3 radar_segmentation_vit4.py     --data-dir ../data/radar_scorp     --num-classes 19     --epochs 15     --batch-size 8     --patch-size 16     --img-size 512 512 --mode train --ckpt-dir checkpoints --load --ckpt ckpt_epoch011.pt


eval:
	root@2d5dc237209e:/workspace/radseg# 
	python3 radar_segmentation_vit4.py     --data-dir ../data/radar_scorp     --num-classes 19     --epochs 10     --batch-size 8     --patch-size 16     --img-size 512 512 --mode eval --ckpt-dir checkpoints --load --save-output --ckpt best.pt
	Eval — pixel_acc: 0.9353, mIoU: 0.3616
	output: checkpoints/preds_eval/
	eval about 10% of the dataset. so pretty fast

infer:
	infer and eval is similar. infer output are saved in seperate folders
	python3 radar_segmentation_vit4.py     --data-dir ../data/radar_scorp     --num-classes 19     --epochs 10     --batch-size 8     --patch-size 16     --img-size 512 512 --mode infer --ckpt-dir checkpoints --load --save-output --ckpt best.pt
	checkpoints/preds_infer/	
	infer all data points. so slower than eval
	 
train results:
Epoch 039: loss=0.2630 acc=0.8818 miou=0.2496
Epoch 040: loss=0.2530 acc=0.8950 miou=0.2546
Epoch 041: loss=0.2459 acc=0.8923 miou=0.2549
Epoch 042: loss=0.2421 acc=0.8952 miou=0.2562
Epoch 043: loss=0.2451 acc=0.9007 miou=0.2593
Epoch 044: loss=0.2378 acc=0.8899 miou=0.2513
Epoch 045: loss=0.2371 acc=0.8918 miou=0.2584
Epoch 046: loss=0.2346 acc=0.8983 miou=0.2506
Epoch 047: loss=0.2277 acc=0.9012 miou=0.2606
Epoch 048: loss=0.2294 acc=0.8980 miou=0.2571
Epoch 049: loss=0.2300 acc=0.9041 miou=0.2706
Epoch 050: loss=0.2210 acc=0.9044 miou=0.2697
