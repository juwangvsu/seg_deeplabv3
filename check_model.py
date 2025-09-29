import torchvision
import torch
from torchvision import models, transforms

m_stk = models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT)
#print(' model stk ', m_stk)

model = torchvision.models.segmentation.deeplabv3_resnet50(
#       weights='DEFAULT', 
        weights=None,
#        weights_backbone=None,
#        num_classes=21
        aux_loss=False
    )
print(' model 2 ', model)

device='cpu'
ckpt_local = torch.load('outputs/epoch_010.ckpt', map_location=device, weights_only=False)

#print('ckpt local key ', ckpt_local['model'].keys())

ckpt_stk = torch.load('model_deeplabv3_pretrained.pt', map_location=device, weights_only=False)
#print('ckpt stk key ', ckpt_stk.keys())
