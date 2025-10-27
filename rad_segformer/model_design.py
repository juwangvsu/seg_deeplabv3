
# model_design.py
from typing import Optional
import torch
import torch.nn as nn

try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except Exception as e:
    raise RuntimeError("Please `pip install transformers`.")

class RadarSegFormer(nn.Module):
    """
    SegFormer configured for single-channel radar input.

    By setting config.num_channels = 1 we create a first conv that accepts 1 channel.
    Note: Pretrained weights (trained on RGB) won't match this shape, so default is training from scratch.
    """
    def __init__(self,
                 num_classes: int,
                 variant: str = 'b2',
                 image_size: int = 512,
                 ignore_index: int = 255,
                 drop_path_rate: float = 0.1,
                 pretrained_name: Optional[str] = None):
        super().__init__()

        # Map simple variant names to standard widths/depths
        variant2dims = {
            'b0': dict(depths=[2, 2, 2, 2], hidden_sizes=[32, 64, 160, 256]),
            'b1': dict(depths=[2, 2, 2, 2], hidden_sizes=[64, 128, 320, 512]),
            'b2': dict(depths=[3, 4, 6, 3], hidden_sizes=[64, 128, 320, 512]),
            'b3': dict(depths=[3, 4, 18, 3], hidden_sizes=[64, 128, 320, 512]),
            'b4': dict(depths=[3, 8, 27, 3], hidden_sizes=[64, 128, 320, 512]),
            'b5': dict(depths=[3, 6, 40, 3], hidden_sizes=[64, 128, 320, 512]),
        }
        dims = variant2dims.get(variant, variant2dims['b2'])

        config = SegformerConfig(
            num_channels=1,
            num_labels=num_classes,
            image_size=image_size,
            ignore_mismatched_sizes=True,
            **dims
        )
        # Set loss ignore index and auxiliary logits behavior
        config.semantic_loss_ignore_index = ignore_index
        config.decoder_hidden_size = 256
        config.drop_path_rate = drop_path_rate

        if pretrained_name is not None:
            # Load a pretrained Segformer and adapt the first conv on-the-fly (may warn about mismatch)
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_name,
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
            # If pretrained had 3 channels, adapt to 1 by averaging weights (simple heuristic)
            first = self.model.segformer.encoder.patch_embeddings[0].proj
            if first.in_channels != 1:
                with torch.no_grad():
                    W = first.weight  # [out, 3, k, k]
                    W = W.mean(dim=1, keepdim=True)  # average across RGB
                    new = nn.Conv2d(1, first.out_channels, kernel_size=first.kernel_size,
                                    stride=first.stride, padding=first.padding, bias=(first.bias is not None))
                    new.weight.copy_(W)
                    if first.bias is not None:
                        new.bias.copy_(first.bias)
                self.model.segformer.encoder.patch_embeddings[0].proj = new
            # Update classifier heads to match num_classes
            self.model.decode_head.classifier = nn.Conv2d(self.model.decode_head.classifier.in_channels,
                                                          num_classes, kernel_size=1)
            if hasattr(self.model, 'auxiliary_head') and self.model.auxiliary_head is not None:
                self.model.auxiliary_head.classifier = nn.Conv2d(self.model.auxiliary_head.classifier.in_channels,
                                                                 num_classes, kernel_size=1)
        else:
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 1, H, W] float
        returns logits: [B, num_classes, H, W]
        """
        outputs = self.model(pixel_values=x)
        return outputs.logits
