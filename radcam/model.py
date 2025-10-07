# pseudocode: radar–camera semantic segmentation with Transformer fusion
# radar format: range–angular image  [B, 1, R, A]
# camera format: RGB image            [B, 3, H, W]

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Positional encodings
# ----------------------------
class Sinusoidal2D(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, h, w, device):
        """
        returns [1, h*w, dim] 2D sine-cosine PE
        """
        y = torch.arange(h, device=device).float()
        x = torch.arange(w, device=device).float()
        yy, xx = torch.meshgrid(y, x, indexing='ij')     # [h,w]
        # split channels for (y,x)
        half = self.dim // 2
        div = torch.exp(torch.arange(half, device=device).float() * (-torch.log(torch.tensor(10000.0))/half))
        pe_y = torch.stack([torch.sin(yy[...,None]*div), torch.cos(yy[...,None]*div)], dim=-1)   # [h,w,half,2]
        pe_x = torch.stack([torch.sin(xx[...,None]*div), torch.cos(xx[...,None]*div)], dim=-1)   # [h,w,half,2]
        pe_y = pe_y.reshape(h, w, half*2)
        pe_x = pe_x.reshape(h, w, half*2)
        pe = torch.cat([pe_y, pe_x], dim=-1)[None]  # [1,h,w,dim*2]; trim if odd
        pe = pe[...,:self.dim].reshape(1, h*w, self.dim)
        return pe

# ----------------------------
# Patch embeddings
# ----------------------------
class PatchEmbed(nn.Module):
    """Generic 2D patchify (Conv) -> tokens"""
    def __init__(self, in_ch, embed_dim, patch=(16,16)):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
    def forward(self, x):
        # x: [B,C,H,W]  ->  tokens: [B, HW, D], fmap: [H', W']
        x = self.proj(x)                      # [B,D,H',W']
        B, D, Hs, Ws = x.shape
        tokens = x.flatten(2).transpose(1,2) # [B,H'*W',D]
        return tokens, (Hs, Ws)

class RadarEncoder(nn.Module):
    """
    Radar range–angular map encoder.
    Treat like an image but with smaller kernels/anisotropic patches if desired.
    """
    def __init__(self, embed_dim, patch=(8,8)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        self.patch = PatchEmbed(32, embed_dim, patch=patch)
    def forward(self, radar):                 # [B,1,R,A]
        x = self.stem(radar)                  # [B,32,R,A]
        tok, hw = self.patch(x)               # [B, Nr, D], (R', A')
        return tok, hw

# ----------------------------
# Transformer blocks
# ----------------------------
class MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden), nn.GELU(),
            nn.Linear(hidden, d)
        )
    def forward(self, x): return self.net(x)

class SelfAttnBlock(nn.Module):
    def __init__(self, d, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn  = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp   = MLP(d, int(d*mlp_ratio))
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        h = self.norm1(x)
        y, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x

class CrossAttnBlock(nn.Module):
    """
    Bidirectional cross-attention + gated residual:
      Cam queries Radar   and   Radar queries Cam
    """
    def __init__(self, d, heads):
        super().__init__()
        self.cam_to_rad = nn.MultiheadAttention(d, heads, batch_first=True)
        self.rad_to_cam = nn.MultiheadAttention(d, heads, batch_first=True)
        self.cam_gate   = nn.Parameter(torch.tensor(0.5))  # learnable scalar gate in [0,1] after sigmoid
        self.rad_gate   = nn.Parameter(torch.tensor(0.5))
        self.cam_norm   = nn.LayerNorm(d)
        self.rad_norm   = nn.LayerNorm(d)
    def forward(self, cam, rad, mask_cam2rad=None, mask_rad2cam=None):
        # normalize
        c = self.cam_norm(cam)
        r = self.rad_norm(rad)
        # cross attention
        c2, _ = self.rad_to_cam(c, r, r, attn_mask=mask_rad2cam, need_weights=False)  # camera queries radar
        r2, _ = self.cam_to_rad(r, c, c, attn_mask=mask_cam2rad, need_weights=False)  # radar  queries camera
        # gated residuals
        gc = torch.sigmoid(self.cam_gate)
        gr = torch.sigmoid(self.rad_gate)
        cam = cam + gc * c2
        rad = rad + gr * r2
        return cam, rad

# ----------------------------
# Fusion backbone
# ----------------------------
class FusionBackbone(nn.Module):
    def __init__(self, d=256, heads=8, depth_cam=4, depth_rad=4, depth_fuse=3):
        super().__init__()
        self.cam_pe  = Sinusoidal2D(d)
        self.rad_pe  = Sinusoidal2D(d)
        self.cam_embed = PatchEmbed(3, d, patch=(16,16))
        self.rad_embed = RadarEncoder(d, patch=(8,8))

        self.cam_blocks = nn.ModuleList([SelfAttnBlock(d, heads) for _ in range(depth_cam)])
        self.rad_blocks = nn.ModuleList([SelfAttnBlock(d, heads) for _ in range(depth_rad)])
        self.fuse_blocks = nn.ModuleList([CrossAttnBlock(d, heads) for _ in range(depth_fuse)])

    def forward(self, img, radar, cam_mask=None, rad_mask=None, cam2rad_mask=None, rad2cam_mask=None):
        """
        img:   [B,3,H,W]
        radar: [B,1,R,A]
        *_mask (optional): sparsity/overlap masks for attention
          - cam2rad_mask: [B, Nc, Nr]   allow which radar tokens each camera token may attend to
          - rad2cam_mask: [B, Nr, Nc]
        returns:
          tokens_cam: [B, Nc, D], fmap_cam (Hc,Wc) for decoding
        """
        # embeddings + 2D PE
        cam_tok, (Hc, Wc) = self.cam_embed(img)        # [B,Nc,D]
        rad_tok, (Hr, Wr) = self.rad_embed(radar)      # [B,Nr,D]
        cam_tok = cam_tok + self.cam_pe(Hc, Wc, img.device)
        rad_tok = rad_tok + self.rad_pe(Hr, Wr, radar.device)

        # modality-specific self-attn stacks
        for blk in self.cam_blocks: cam_tok = blk(cam_tok, key_padding_mask=cam_mask)
        for blk in self.rad_blocks: rad_tok = blk(rad_tok, key_padding_mask=rad_mask)

        # cross-attention fusion stack
        for blk in self.fuse_blocks:
            cam_tok, rad_tok = blk(cam_tok, rad_tok, mask_cam2rad=cam2rad_mask, mask_rad2cam=rad2cam_mask)

        return cam_tok, (Hc, Wc), rad_tok, (Hr, Wr)

# ----------------------------
# Segmentation decoder + auxiliary radar head
# ----------------------------
class SegDecoder(nn.Module):
    def __init__(self, d, num_classes, fmap_hw):
        super().__init__()
        # reshape tokens -> [B,D,Hc,Wc] then upsample to image space
        self.num_classes = num_classes
        self.fmap_hw = fmap_hw  # (Hc,Wc) only used in forward
        self.conv = nn.Sequential(
            nn.Conv2d(d, d, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(d, d//2, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(d//2, num_classes, 1)

    def forward(self, cam_tokens, target_size):
        B, Nc, D = cam_tokens.shape
        Hc, Wc = self.fmap_hw
        x = cam_tokens.transpose(1,2).reshape(B, D, Hc, Wc)        # [B,D,Hc,Wc]
        x = self.conv(x)
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        logits = self.head(x)                                      # [B,K,H,W]
        return logits

class AuxRadarHead(nn.Module):
    """Optional: objectness / velocity prediction on radar tokens for stabilizing training."""
    def __init__(self, d, out_dim=2):
        super().__init__()
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, out_dim))
    def forward(self, rad_tokens):  # [B,Nr,D]
        return self.head(rad_tokens)  # [B,Nr,out_dim]

# ----------------------------
# Full model
# ----------------------------
class RadarCameraSeg(nn.Module):
    def __init__(self, num_classes=19, d=256, heads=8):
        super().__init__()
        self.backbone = FusionBackbone(d=d, heads=heads)
        # decoder Hc,Wc will be known after first forward; keep placeholder
        self.seg_decoder = None
        self.aux_radar = AuxRadarHead(d, out_dim=2)  # e.g., objectness, radial-vel bin

    def forward(self, img, radar, overlap_masks=None):
        """
        overlap_masks (optional dict) may include:
          cam_mask: [B,Nc]           (token padding)
          rad_mask: [B,Nr]
          cam2rad:  [B,Nc,Nr]        (restrict cam tokens to attend to plausible radar cells)
          rad2cam:  [B,Nr,Nc]
        """
        cam_tok, (Hc,Wc), rad_tok, (Hr,Wr) = self.backbone(
            img, radar,
            cam_mask = overlap_masks.get('cam_mask') if overlap_masks else None,
            rad_mask = overlap_masks.get('rad_mask') if overlap_masks else None,
            cam2rad_mask = overlap_masks.get('cam2rad') if overlap_masks else None,
            rad2cam_mask = overlap_masks.get('rad2cam') if overlap_masks else None,
        )

        if self.seg_decoder is None:
            self.seg_decoder = SegDecoder(cam_tok.size(-1), num_classes=self.num_classes, fmap_hw=(Hc,Wc))
        seg_logits = self.seg_decoder(cam_tok, target_size=img.shape[-2:])  # [B,K,H,W]

        aux_logits = self.aux_radar(rad_tok)                                # [B,Nr,2]
        return seg_logits, aux_logits

    @property
    def num_classes(self):
        return self.seg_decoder.num_classes if self.seg_decoder else None

# ----------------------------
# Utility: building sparsity masks (sketch)
# ----------------------------
def build_overlap_masks(calib, Hc, Wc, Hr, Wr, B, device):
    """
    Sketch: use calibration to map camera tokens (Hc×Wc) ↔ radar cells (Hr×Wr).
    Return boolean masks with False = allowed, True = disallowed for nn.MultiheadAttention.
    """
    Nc, Nr = Hc*Wc, Hr*Wr
    cam2rad = torch.ones(B, Nc, Nr, dtype=torch.bool, device=device)
    rad2cam = torch.ones(B, Nr, Nc, dtype=torch.bool, device=device)
    # Fill cam2rad[b,i,j]=False if (cam token i) overlaps radar cell j given extrinsics/intrinsics + range-angle model.
    # (Left as implementation-specific)
    return dict(cam2rad=cam2rad, rad2cam=rad2cam)

# ----------------------------
# Training tips (not code)
# ----------------------------
# • Modality dropout: with p≈0.1 zero-out radar or image tokens per batch.
# • Balance gradients: loss = CE(seg) + λ * BCE(aux_radar); start λ≈0.3.
# • Curriculum: train self-attn blocks first; enable cross-attn blocks after N epochs.
# • Efficiency: use windowed attention for cam tokens; keep radar tokens dense (Nr<<Nc).

