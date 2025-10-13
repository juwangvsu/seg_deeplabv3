import timm
from typing import List, Tuple
import torch.nn as nn

try:
    from mit import MixVisionTransformer, MIT_CONFIGS
except Exception:
    MixVisionTransformer = None
    MIT_CONFIGS = {}

_DEFAULT_CHANNELS = {
    "pvt_v2_b0": [32, 64, 160, 256],
    "pvt_v2_b1": [64, 128, 320, 512],
    "pvt_v2_b2": [64, 128, 320, 512],
    "pvt_v2_b3": [64, 128, 320, 512],
    "pvt_v2_b4": [64, 128, 320, 512],
    "pvt_v2_b5": [64, 128, 320, 512],
}

def _create_timm_backbone(name: str, pretrained: bool = True, out_indices=(0, 1, 2, 3)):
    model = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=out_indices)
    info = model.feature_info
    channels = [f["num_chs"] for f in info]
    reduction = [f["reduction"] for f in info]
    if not channels or not reduction:
        channels = _DEFAULT_CHANNELS.get(name)
        reduction = [4, 8, 16, 32]
        if channels is None:
            raise ValueError(f"Unknown channels for backbone {name}. Please add to _DEFAULT_CHANNELS.")
    return model, channels, reduction

def _create_mit_backbone(name: str):
    assert MixVisionTransformer is not None, "mit.py not found — ensure it exists."
    if name not in MIT_CONFIGS:
        raise ValueError(f"Unknown MiT variant '{name}'. Options: {list(MIT_CONFIGS.keys())}")
    cfg = MIT_CONFIGS[name]
    model = MixVisionTransformer(**cfg)
    channels = cfg["embed_dims"]
    reduction = [4, 8, 16, 32]
    return model, channels, reduction

def create_pyramid_backbone(name: str = "pvt_v2_b2", pretrained: bool = True, out_indices=(0, 1, 2, 3)):
    if name.startswith("mit_b"):
        return _create_mit_backbone(name)
    else:
        return _create_timm_backbone(name, pretrained=pretrained, out_indices=out_indices)
