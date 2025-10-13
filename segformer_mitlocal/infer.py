import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

from backbones import create_pyramid_backbone
from segformer import SegFormer
from palettes import CITYSCAPES_PALETTE

def colorize(mask: np.ndarray, palette):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for i, (r, g, b) in enumerate(palette):
        out[mask == i] = (r, g, b)
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="mit_b2")
    parser.add_argument("--num-classes", type=int, default=19)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--images", type=str, required=True)
    parser.add_argument("--out", type=str, default="out")
    parser.add_argument("--img-size", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bk, channels, _ = create_pyramid_backbone(args.backbone, pretrained=False)
    model = SegFormer(bk, channels, num_classes=args.num_classes, decoder_embed_dim=256)
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd.get("model", sd))
    model.to(device).eval()

    out_dir = Path(args.out)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    (out_dir / "color").mkdir(parents=True, exist_ok=True)
    (out_dir / "overlay").mkdir(parents=True, exist_ok=True)

    tfm = T.Compose([
        T.Resize((args.img_size, args.img_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    for p in sorted(Path(args.images).glob("*")):
        img = Image.open(p).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=="cuda"):
            logits = model(x)
            pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

        Image.fromarray(pred).save(out_dir / "masks" / (p.stem + ".png"))
        color = colorize(pred, CITYSCAPES_PALETTE)
        Image.fromarray(color).save(out_dir / "color" / (p.stem + ".png"))
        base = np.array(img.resize((pred.shape[1], pred.shape[0]), Image.BILINEAR))
        overlay = (0.4 * base + 0.6 * color).astype(np.uint8)
        Image.fromarray(overlay).save(out_dir / "overlay" / (p.stem + ".png"))

if __name__ == "__main__":
    main()
