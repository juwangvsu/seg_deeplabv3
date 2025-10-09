import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

try:
    import timm
except ImportError as e:
    raise RuntimeError("Please install timm (pip install timm)")

class SegFormerHead(nn.Module):
    '''
    Lightweight all-MLP decoder head (implemented with 1x1 Convs).
    '''
    def __init__(self, in_channels: List[int], embedding_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.proj_convs = nn.ModuleList([nn.Conv2d(c, embedding_dim, kernel_size=1) for c in in_channels])
        self.dropout = nn.Dropout2d(dropout)
        self.fuse_mlp = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
        self.classifier = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]):
        x1, x2, x3, x4 = feats
        _, _, H, W = x1.shape
        outs = []
        for x, conv in zip([x1, x2, x3, x4], self.proj_convs):
            y = conv(x)
            if y.shape[-2:] != (H, W):
                y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=False)
            outs.append(y)
        y = torch.cat(outs, dim=1)
        y = self.fuse_mlp(y)
        y = self.dropout(y)
        logits = self.classifier(y)
        return logits

class SegFormer(nn.Module):
    def __init__(self, num_classes: int = 19, encoder_variant: str = 'pvt_v2_b2', embed_dim: int = 256):
        super().__init__()
        avail = timm.list_models('pvt*')
        if encoder_variant not in avail:
            raise RuntimeError(f"Unknown encoder '{encoder_variant}'. Try one of: {avail}")
        self.backbone = timm.create_model(
            encoder_variant, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3)
        )
        in_channels = self.backbone.feature_info.channels()
        if len(in_channels) != 4:
            raise RuntimeError(f'Backbone returned {len(in_channels)} feature maps, expected 4. Got channels: {in_channels}')
        self.decode_head = SegFormerHead(in_channels=in_channels, embedding_dim=embed_dim, num_classes=num_classes)

    def forward(self, x: torch.Tensor):
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        logits = F.interpolate(logits, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return logits
