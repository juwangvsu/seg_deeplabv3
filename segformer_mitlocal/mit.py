from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans: int, embed_dim: int, patch_size: int, stride: int, padding: int):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Conv2d(dim, hidden, 1)
        self.dwconv = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, sr_ratio: int = 1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.q = nn.Conv2d(dim, dim, 1)
        self.kv = nn.Conv2d(dim, dim * 2, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.sr_ratio = sr_ratio
        self.sr = None
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = LayerNorm2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q = self.q(x)
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W).permute(0, 1, 3, 2)

        if self.sr is not None:
            x_ = self.sr(x)
            x_ = self.norm(x_)
        else:
            x_ = x
        kv = self.kv(x_)
        k, v = kv.chunk(2, dim=1)
        HpW = k.shape[-2] * k.shape[-1]
        k = k.reshape(B, self.num_heads, C // self.num_heads, HpW).permute(0, 1, 3, 2)
        v = v.reshape(B, self.num_heads, C // self.num_heads, HpW).permute(0, 1, 3, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 1, 3, 2).reshape(B, C, H, W)
        x = self.proj(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, sr_ratio: int, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = LayerNorm2d(dim)
        self.attn = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.norm2 = LayerNorm2d(dim)
        self.mlp = MLP(dim, mlp_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class MixVisionTransformer(nn.Module):
    def __init__(
        self,
        embed_dims: List[int],
        depths: List[int],
        num_heads: List[int],
        mlp_ratio: float = 4.0,
        sr_ratios: List[int] = (8, 4, 2, 1),
        in_chans: int = 3,
        drop_path_rate: float = 0.1,
    ):
        super().__init__()
        assert len(embed_dims) == len(depths) == len(num_heads) == len(sr_ratios) == 4

        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(in_chans, embed_dims[0], patch_size=7, stride=4, padding=3),
            OverlapPatchEmbed(embed_dims[0], embed_dims[1], patch_size=3, stride=2, padding=1),
            OverlapPatchEmbed(embed_dims[1], embed_dims[2], patch_size=3, stride=2, padding=1),
            OverlapPatchEmbed(embed_dims[2], embed_dims[3], patch_size=3, stride=2, padding=1),
        ])

        dpr = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(Block(embed_dims[i], num_heads[i], mlp_ratio, sr_ratios[i], drop_path=dpr[cur + j]))
            cur += depths[i]
            self.stages.append(nn.Sequential(*blocks))

    def forward(self, x):
        feats = []
        for i in range(4):
            x = self.patch_embeds[i](x)
            x = self.stages[i](x)
            feats.append(x)
        return feats

MIT_CONFIGS = {
    "mit_b0": dict(embed_dims=[32, 64, 160, 256], depths=[2, 2, 2, 2], num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit_b1": dict(embed_dims=[64, 128, 320, 512], depths=[2, 2, 2, 2], num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit_b2": dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 6, 3],  num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit_b3": dict(embed_dims=[64, 128, 320, 512], depths=[3, 4, 18, 3], num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit_b4": dict(embed_dims=[64, 128, 320, 512], depths=[3, 8, 27, 3], num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
    "mit_b5": dict(embed_dims=[64, 128, 320, 512], depths=[3, 6, 40, 3], num_heads=[1, 2, 5, 8], sr_ratios=[8, 4, 2, 1]),
}
