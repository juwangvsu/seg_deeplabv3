import sys
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50

RTSP_URL = "rtsp://150.174.3.15:80/mystream"

# -------------------------
# Model setup
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = deeplabv3_resnet50(weights="DEFAULT")  # torchvision >= 0.13
model.to(device)
model.eval()

# Image transforms (normalize like ImageNet)
preprocess = T.Compose([
    T.ToPILImage(),
    T.Resize(520),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# Simple color map for classes
def create_pascal_label_colormap():
    import numpy as np
    colormap = np.zeros((256, 3), dtype=int)
    for i in range(256):
        r = g = b = 0
        cid = i
        for j in range(8):
            r |= (((cid >> 0) & 1) << (7 - j))
            g |= (((cid >> 1) & 1) << (7 - j))
            b |= (((cid >> 2) & 1) << (7 - j))
            cid >>= 3
        colormap[i] = [r, g, b]
    return colormap

colormap = create_pascal_label_colormap()

def overlay_segmentation(frame_bgr, seg_mask):
    """Overlay colored segmentation mask on top of frame."""
    import numpy as np

    h, w = frame_bgr.shape[:2]
    seg_mask = cv2.resize(seg_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    color_mask = colormap[seg_mask].astype("uint8")  # (H, W, 3) in RGB
    color_mask_bgr = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

    alpha = 0.5
    overlaid = cv2.addWeighted(frame_bgr, 1 - alpha, color_mask_bgr, alpha, 0)
    return overlaid

# -------------------------
# Video capture
# -------------------------
cap = cv2.VideoCapture(RTSP_URL)

if not cap.isOpened():
    print(f"Failed to open RTSP stream: {RTSP_URL}", file=sys.stderr)
    sys.exit(1)

# Read one frame to get size for ffplay
ret, frame = cap.read()
if not ret:
    print("Could not read initial frame", file=sys.stderr)
    sys.exit(1)

print(" read initial frame", ret, frame.shape) 
height, width = frame.shape[:2]

# Write the first frame later in the loop (we'll process it too)
#cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# -------------------------
# Main loop
# -------------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        #print(f"get a frame {ret} {frame}", file=sys.stderr)
        # Run segmentation
        with torch.no_grad():
            inp = preprocess(frame).unsqueeze(0).to(device)
            out = model(inp)["out"]  # [1, C, H, W]
            seg = out.argmax(1).squeeze(0).cpu().numpy().astype("uint8")

        # Overlay
        overlaid = overlay_segmentation(frame, seg)

        # Write raw BGR24 bytes to stdout for ffplay
        #print(f"frame seg rst {seg}", file=sys.stderr)
        sys.stdout.buffer.write(overlaid.tobytes())
        sys.stdout.flush()

except KeyboardInterrupt:
    pass
finally:
    cap.release()

