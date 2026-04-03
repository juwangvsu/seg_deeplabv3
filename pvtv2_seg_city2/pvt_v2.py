"""
Minimal, production-ready Pyramid Vision Transformer v2 (PVTv2)
with SegHeadLite and SegModel wrapper.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int, stride: int, padding: int = None):
        super().__init__()
        if padding is None:
            padding = patch_size // 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W

class DWConv(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dw(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class MLPWithDWConv(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.dw = DWConv(hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = self.fc1(x); x = self.act(x); x = self.dw(x, H, W); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class SRABlock(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int, qkv_bias: bool=True, attn_drop: float=0.0, proj_drop: float=0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.d = dim // heads
        self.scale = self.d ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio, groups=dim)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
            self.norm = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.heads, self.d).transpose(1, 2)
        if self.sr is not None:
            x_ = x.transpose(1, 2).view(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.heads, self.d).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.heads, self.d).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PVTv2Block(nn.Module):
    def __init__(self, dim: int, heads: int, sr_ratio: int, mlp_ratio: float, drop: float, attn_drop: float, drop_path: float):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = SRABlock(dim, heads, sr_ratio, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        self.dp = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.n2 = nn.LayerNorm(dim)
        self.mlp = MLPWithDWConv(dim, mlp_ratio, drop)
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        x = x + self.dp(self.attn(self.n1(x), H, W))
        x = x + self.dp(self.mlp(self.n2(x), H, W))
        return x

class PVTv2Stage(nn.Module):
    def __init__(self, in_ch, embed_dim, depth, heads, patch, stride, sr_ratio, mlp_ratio, drop, attn_drop, dpr_list):
        super().__init__()
        self.patch = OverlapPatchEmbed(in_ch, embed_dim, patch, stride)
        self.blocks = nn.ModuleList([PVTv2Block(embed_dim, heads, sr_ratio, mlp_ratio, drop, attn_drop, dpr_list[i]) for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x: torch.Tensor):
        x, H, W = self.patch(x)
        for b in self.blocks:
            x = b(x, H, W)
        x = self.norm(x)
        B, N, C = x.shape
        feat = x.transpose(1, 2).view(B, C, H, W)
        return feat, H, W

from dataclasses import dataclass
@dataclass
class PVTv2Config:
    embed_dims: Tuple[int,int,int,int]
    depths: Tuple[int,int,int,int]
    num_heads: Tuple[int,int,int,int]
    sr_ratios: Tuple[int,int,int,int]
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.1
    in_chans: int = 3

class PVTv2(nn.Module):
    def __init__(self, cfg: PVTv2Config):
        super().__init__()
        self.cfg = cfg
        total = sum(cfg.depths)
        dpr = torch.linspace(0, cfg.drop_path_rate, total).tolist()
        it = iter(dpr)
        in_ch = cfg.in_chans
        patch = (7,3,3,3); stride=(4,2,2,2)
        stages = []
        for i in range(4):
            depth = cfg.depths[i]
            dpr_i = [next(it) for _ in range(depth)]
            st = PVTv2Stage(in_ch, cfg.embed_dims[i], depth, cfg.num_heads[i], patch[i], stride[i], cfg.sr_ratios[i], cfg.mlp_ratio, cfg.drop_rate, cfg.attn_drop_rate, dpr_i)
            stages.append(st); in_ch = cfg.embed_dims[i]
        self.stages = nn.ModuleList(stages)
    @property
    def out_channels(self): return self.cfg.embed_dims
    def forward(self, x: torch.Tensor):
        feats = []
        for st in self.stages:
            x, H, W = st.patch(x)
            B, N, C = x.shape
            for b in st.blocks:
                x = b(x, H, W)
            x = st.norm(x)
            x = x.transpose(1, 2).view(B, C, H, W)
            feats.append(x)
        return feats

def _cfg(embed, depths, heads, dr): 
    return PVTv2Config(embed_dims=embed, depths=depths, num_heads=heads, sr_ratios=(8,4,2,1), drop_path_rate=dr)
def pvt_v2_b0(**kw): return PVTv2(_cfg((32,64,160,256),(2,2,2,2),(1,2,5,8),0.1))
def pvt_v2_b1(**kw): return PVTv2(_cfg((64,128,320,512),(2,2,2,2),(1,2,5,8),0.1))
def pvt_v2_b2(**kw): return PVTv2(_cfg((64,128,320,512),(3,4,6,3),(1,2,5,8),0.15))
def pvt_v2_b3(**kw): return PVTv2(_cfg((64,128,320,512),(3,4,18,3),(1,2,5,8),0.2))
def pvt_v2_b4(**kw): return PVTv2(_cfg((64,128,320,512),(3,8,27,3),(1,2,5,8),0.3))
def pvt_v2_b5(**kw): return PVTv2(_cfg((64,128,320,512),(3,6,40,3),(1,2,5,8),0.35))

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class SegHeadLite(nn.Module):
    def __init__(self, in_channels: Tuple[int,int,int,int], num_classes: int, width: int = 128):
        super().__init__()
        c2, c3, c4, c5 = in_channels
        self.l2 = ConvBNAct(c2, width, 1)
        self.l3 = ConvBNAct(c3, width, 1)
        self.l4 = ConvBNAct(c4, width, 1)
        self.l5 = ConvBNAct(c5, width, 1)
        #self.smooth = ConvBNAct(width, width, 3, 1, 1)
        self.cls = nn.Conv2d(width, num_classes, 1)
        self.smooth = ConvBNAct(width, width, 3, 1, 1)
    def forward(self, feats: list, input_size: Tuple[int,int]):
        f2, f3, f4, f5 = feats
        print(f"f2.shape {f2.shape}, f3.shape {f3.shape} f4.shape {f4.shape} f5.shape {f5.shape}")
        h, w = f2.shape[-2:]
        p2 = self.l2(f2)
        p3 = self.l3(f3)
        p4 = self.l4(f4)
        p5 = self.l5(f5)
        print(f"p2.shape {p2.shape}, .shape {p3.shape} 4.shape {p4.shape} 5.shape {p5.shape}")
        p3 = F.interpolate(p3, size=(h,w), mode="bilinear", align_corners=False)

        p4 = F.interpolate(p4, size=(h,w), mode="bilinear", align_corners=False)
        p5 = F.interpolate(p5, size=(h,w), mode="bilinear", align_corners=False)

        x = self.smooth(p2 + p3 + p4 + p5)
        logits = self.cls(x)
        rst= F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)
        #print('xxx x.shape, logits.shape, F.inte(logits).shape', x.shape, logits.shape, rst.shape)
        return rst

class SegModel(nn.Module):
    def __init__(self, backbone: Optional[nn.Module]=None, variant: str="b2", num_classes: int=19):
        super().__init__()
        if backbone is None:
            builder = {"b0":pvt_v2_b0,"b1":pvt_v2_b1,"b2":pvt_v2_b2,"b3":pvt_v2_b3,"b4":pvt_v2_b4,"b5":pvt_v2_b5}[variant.lower()]
            backbone = builder()
        self.backbone = backbone
        self.head = SegHeadLite(self.backbone.out_channels, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        rst= self.head(feats, input_size=x.shape[-2:])
        #print('xxx feats len, .shape rst.shape', len(feats) , feats[0].shape, feats[1].shape, feats[2].shape, feats[3].shape, rst.shape)
        return rst
