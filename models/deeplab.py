import torch
import torch.nn as nn
import torchvision

def create_deeplabv3_resnet50(num_classes=5, pretrained_backbone=True):
    model = torchvision.models.segmentation.deeplabv3_resnet50(
        weights=None,
        weights_backbone=torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None,
        num_classes=num_classes
    )
    return model
