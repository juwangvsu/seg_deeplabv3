import os, argparse, glob
import numpy as np
import cv2
import torch
from models.deeplab import create_deeplabv3_resnet50

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    #print('xxx ', img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2,0,1))
    return img

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--num_classes", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    model = create_deeplabv3_resnet50(num_classes=args.num_classes, pretrained_backbone=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    for img_path in sorted(glob.glob(os.path.join(args.input_dir, "*"))):
        img = load_image(img_path)
        x = torch.from_numpy(img).unsqueeze(0).to(device)
        out = model(x)["out"]
        pred = torch.argmax(out, dim=1)[0].cpu().numpy().astype(np.uint8)
        # Save raw ID mask
        base = os.path.splitext(os.path.basename(img_path))[0]
        cv2.imwrite(os.path.join(args.output_dir, f"{base}_mask.png"), pred)

if __name__ == "__main__":
    main()
