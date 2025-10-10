import os
import argparse
from glob import glob
from PIL import Image
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor, normalize

from segformer import SegFormer
from utils import colorize_mask, overlay_image

def load_image(path):
    img = Image.open(path).convert('RGB')
    return img

def preprocess(img, input_size):
    w, h = img.size
    scale = input_size / min(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img = img.resize((nw, nh), Image.BILINEAR)
    i = max(0, (nh - input_size) // 2)
    j = max(0, (nw - input_size) // 2)
    img_crop = img.crop((j, i, j + input_size, i + input_size))
    x = to_tensor(img_crop)
    x = normalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return x.unsqueeze(0), img_crop

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help='glob pattern or folder of images')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs_infer')
    parser.add_argument('--num_classes', type=int, default=19)
    parser.add_argument('--encoder', type=str, default='pvt_v2_b2')
    parser.add_argument('--input_size', type=int, default=768)
    parser.add_argument('--showmodel', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SegFormer(num_classes=args.num_classes, encoder_variant=args.encoder)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'], strict=False)
    print(model)
    if args.showmodel:
        exit(0)
    model.to(device)
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    mask_dir = os.path.join(args.out_dir, 'masks')
    color_dir = os.path.join(args.out_dir, 'color_mask')
    overlay_dir = os.path.join(args.out_dir, 'overlay')
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    paths = []
    if os.path.isdir(args.images):
        for ext in ('*.png', '*.jpg', '*.jpeg'):
            paths.extend(glob(os.path.join(args.images, ext)))
    else:
        paths = glob(args.images)
    assert len(paths) > 0, f"No images found for {args.images}"

    with torch.no_grad():
        for p in paths:
            img = load_image(p)
            x, img_crop = preprocess(img, args.input_size)
            x = x.to(device)
            logits = model(x)
            pred = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

            base = os.path.splitext(os.path.basename(p))[0]
            Image.fromarray(pred, mode='L').save(os.path.join(mask_dir, base + '.png'))
            color = colorize_mask(pred)
            color.save(os.path.join(color_dir, base + '.png'))
            over = overlay_image(img_crop, color, alpha=0.5)
            over.save(os.path.join(overlay_dir, base + '.png'))
            print(f"Saved {base}.png")

if __name__ == '__main__':
    main()
