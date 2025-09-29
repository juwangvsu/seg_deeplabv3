import torch
import torch.nn as nn
import torchvision

def create_deeplabv3_resnet50(num_classes=21, pretrained_backbone=True):
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None,
        weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None,
        aux_loss=True,
        num_classes=num_classes
    )
    return model
