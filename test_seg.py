from PIL import Image
import matplotlib.pyplot as plt
import torch
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

# DeepLabv3 model finetuned on ADE20K (150 classes)
model_name = "facebook/detr-resnet-50-panoptic"  # Panoptic example (alt: nvidia/segformer-b0-finetuned-ade-512-512)
# For pure DeepLabv3 you can use: "Intel/dpt-large-ade" (depth seg) or
# "mattmdjaga/segformer_b0_clothes" as another example; choose an ADE20K/Cityscapes model you prefer.

processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = AutoModelForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

img = Image.open("000010.png").convert("RGB")
inputs = processor(images=img, return_tensors="pt")
with torch.no_grad():
        outputs = model(**inputs)
        # logits shape: [1, num_labels, h/4, w/4] typically
        upsampled = torch.nn.functional.interpolate(
                    outputs.logits,
                        size=img.size[::-1],
                            mode="bilinear",
                                align_corners=False,
                                )
        pred = upsampled.argmax(1)[0].cpu()
        print('xxx' , pred)
        plt.figure(figsize=(16,6))
        plt.subplot(1,2,1); plt.imshow(img); plt.title("Original"); plt.axis("off")
        plt.subplot(1,2,2); plt.imshow(pred, cmap="nipy_spectral"); plt.title("Semantic Segmentation"); plt.axis("off")
        plt.tight_layout(); plt.show()

