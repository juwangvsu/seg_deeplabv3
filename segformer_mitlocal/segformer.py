from typing import List
import torch
import torch.nn as nn

class MLPDecoderHead(nn.Module):
    def __init__(self, in_channels: List[int], embed_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        assert len(in_channels) == 4, "Expecting 4 pyramid features"
        self.proj = nn.ModuleList([nn.Sequential(
            nn.Conv2d(c, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        ) for c in in_channels])

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 4, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        size = feats[0].shape[-2:]
        outs = []
        for i, f in enumerate(feats):
            p = self.proj[i](f)
            if p.shape[-2:] != size:
                p = torch.nn.functional.interpolate(p, size=size, mode="bilinear", align_corners=False)
            outs.append(p)
        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

class SegFormer(nn.Module):
    def __init__(self, backbone: nn.Module, feature_channels: List[int], num_classes: int,
                 decoder_embed_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.backbone = backbone
        self.decode_head = MLPDecoderHead(feature_channels, decoder_embed_dim, num_classes, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        logits = torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits
