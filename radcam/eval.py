import os
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from pathlib import Path
from radcam.dataset import RadarCamSegDataset
# reverse of ImageNet normalization used in dataset
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def denormalize_img(t):  # t: [3,H,W], torch
    x = t.detach().float().cpu().numpy()
    x = (x.transpose(1,2,0) * IMAGENET_STD + IMAGENET_MEAN).clip(0,1)
    x = (x * 255.0).round().astype(np.uint8)  # [H,W,3]
    return Image.fromarray(x)

def build_palette(num_classes):
    # deterministic distinct colors
    rng = np.random.RandomState(123)
    palette = rng.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    palette[0] = np.array([0, 0, 0], dtype=np.uint8)   # background = black
    return palette

def colorize_mask(mask, palette):  # mask: [H,W] int
    h, w = mask.shape
    c = palette[mask.clip(0, len(palette)-1)]
    return Image.fromarray(c, mode='RGB')
from PIL import Image

def overlay_image(rgb_pil: Image.Image, color_mask_pil: Image.Image, alpha: float = 0.45) -> Image.Image:
    """
    Overlay a color segmentation mask on top of an RGB image using PIL.Image.blend.

    Args:
        rgb_pil (PIL.Image.Image): Base RGB image.
        color_mask_pil (PIL.Image.Image): Colorized segmentation mask (RGB).
        alpha (float): Blend factor (0 → only rgb, 1 → only color mask).

    Returns:
        PIL.Image.Image: Overlay image.
    """
    # Ensure both are RGB mode and same size
    if rgb_pil.mode != "RGB":
        rgb_pil = rgb_pil.convert("RGB")
    if color_mask_pil.mode != "RGB":
        color_mask_pil = color_mask_pil.convert("RGB")

    if rgb_pil.size != color_mask_pil.size:
        color_mask_pil = color_mask_pil.resize(rgb_pil.size, resample=Image.NEAREST)

    overlay = Image.blend(rgb_pil, color_mask_pil, alpha)
    return overlay

@torch.no_grad()
def run_eval(model, image_dir, mask_dir, radar_dir, out_dir, num_classes=19, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    # dirs
    out_dir = Path(out_dir)
    d_orig  = out_dir / 'orig'
    d_mask  = out_dir / 'masks'
    d_colo  = out_dir / 'color_mask'
    d_ovl   = out_dir / 'overlay'
    for d in [d_orig, d_mask, d_colo, d_ovl]: d.mkdir(parents=True, exist_ok=True)

    # loader
    ds = RadarCamSegDataset(image_dir=image_dir, mask_dir=mask_dir, radar_dir=radar_dir, num_classes=num_classes)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    palette = build_palette(num_classes)

    pbar = tqdm(dl, desc='Eval', unit='img')
    for batch in pbar:
        img   = batch['image'].to(device, non_blocking=True).float()   # [1,3,H,W]
        radar = batch['radar'].to(device, non_blocking=True).float()   # [1,1,R,A]
        stem = batch['stem'][0] # batch size 1 so ok
        print('xxx ', stem)
        # forward
        # NOTE: model already upsamples to image size inside decoder as designed
        seg_logits, _ = model(img, radar, overlap_masks=None)  # [1,K,H,W]
        pred = seg_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)  # [H,W]
        print(' yyy img.shape ', img.shape)
        img_p = os.path.join(image_dir, stem+".png")
        print(' yyy img_p ', img_p)
        orig = Image.open(img_p).convert('RGB')
        #orig = Image.fromarray(torch.squeeze(img.cpu(), dim=0).numpy().astype(np.uint8), mode='RGB')
        orig.save(d_orig / f'{stem}.png')
        # save grayscale mask (H,W)
        m_pil = Image.fromarray(pred, mode='L')
        m_pil.save(d_mask / f'{stem}.png')

        # save color mask
        colored_mask_image = Image.fromarray(pred).resize(orig.size, Image.NEAREST)
        cm_pil = colorize_mask(pred, palette)
        cm_pil = cm_pil.resize(orig.size, resample=Image.NEAREST)  # match original for overlay
        cm_pil.save(d_colo / f'{stem}.png')
        #colored_mask_image.save(d_colo / f'{stem}_a.png')

        # save overlay
        ov_pil = overlay_image(orig, cm_pil, alpha=0.45)
        ov_pil.save(d_ovl / f'{stem}.jpg')
        #ov_pil = overlay_image(orig, colored_mask_image, alpha=0.45)
        #ov_pil.save(d_ovl / f'{stem}_a.jpg')

