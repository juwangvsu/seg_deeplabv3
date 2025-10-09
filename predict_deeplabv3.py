import torch
from torchvision import models, transforms
from PIL import Image
import os
import argparse
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
# load stock pretrained and dump to local file deepv3 resnet50
# --load_ckpt load a local pt file, this override the default one
# --save_ckpt save to a local pt file

def create_deeplabv3_model(model_type: str):
    """Loads a pretrained DeepLabV3 model with the specified backbone."""
    if model_type == 'resnet101':
        return models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT).eval()
    elif model_type == 'resnet50':
        return models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT).eval()
    else:
        raise ValueError("Invalid model_type. Choose 'resnet50' or 'resnet101'.")
import colorsys

def build_palette(n: int, seed: int = 0) -> np.ndarray:
    """Deterministic [n,3] RGB palette (0..255)."""
    rng = np.random.default_rng(seed)
    hues = np.linspace(0, 1, n, endpoint=False)
    sat = 0.75
    val = 0.95
    colors = []
    for h in hues:
        r, g, b = colorsys.hsv_to_rgb(float(h), sat, val)
        colors.append([int(r * 255), int(g * 255), int(b * 255)])
    colors = np.array(colors, dtype=np.uint8)
    return colors[rng.permutation(n)]

def get_color_palette():
    """Returns a color palette for the 21 classes (including background)."""
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    return (colors % 255).numpy().astype("uint8")

def create_overlay(original_image, colored_mask, alpha=0.5):
    """Blends the original image and the colored mask."""
    overlay = Image.blend(original_image.convert("RGB"), colored_mask.convert("RGB"), alpha)
    return overlay

def load_labels(model) -> List[str]:
    """Get contiguous id->label list from HF model config."""
    weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    labels = weights.meta["categories"]

    # Print the list of labels
    print(labels)
    '''
    id2label_raw: Dict[str, str] = getattr(model.config, "id2label", {})
    if id2label_raw:
        id2label = {int(k): v for k, v in id2label_raw.items()}
        num_labels = max(id2label.keys()) + 1
        labels = [id2label.get(i, f"class_{i}") for i in range(num_labels)]
    else:
        num_labels = getattr(model.config, "num_labels", 150)
        labels = [f"class_{i}" for i in range(num_labels)]a
    '''
    return labels

def save_overlay(img, blend, overlay_out,palette, no_legend=False, labels=None, show_ids=[1,2]):
    #show_ids=[1,2]
    # Plot and save overlay PNG
    plt.figure(figsize=(16, 7))
    ax1 = plt.subplot(1, 2, 1)
    ax1.imshow(img)
    ax1.set_title("Original")
    ax1.axis("off")
    ax2 = plt.subplot(1, 2, 2)
    ax2.imshow(blend)
    ax2.set_title("Segmentation (overlay)")
    ax2.axis("off")
    print('yyy ')
    if not no_legend and len(show_ids) > 0:
        legend_patches = [
            Patch(facecolor=palette[i] / 255.0, edgecolor="black", label=f"{i}: {labels[i]}")
            for i in show_ids
        ]
        ax2.legend(
            handles=legend_patches,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            borderaxespad=0.0,
            frameon=True,
            title="Classes",
        )

    plt.tight_layout()
    plt.savefig(overlay_out, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"    overlay:      {overlay_out}")

def segment_and_save_images(input_dir: str, raw_mask_dir: str, colored_mask_dir: str, overlay_dir: str, model_type: str = 'resnet50', load_ckpt=False, save_ckpt=False, ckptfn="model_deeplabv3_pretrained.pt", filenamex="", quickone=False):
    """
    Performs semantic segmentation on all images in a folder and saves
    raw masks, colored masks, and overlayed images to specified directories.

    Args:
        input_dir (str): Path to the directory containing input images.
        raw_mask_dir (str): Path to the directory for saving raw grayscale masks.
        colored_mask_dir (str): Path to the directory for saving colored masks.
        overlay_dir (str): Path to the directory for saving overlayed images.
        model_type (str): The ResNet backbone to use ('resnet50' or 'resnet101').
    """
    # Create output directories if they don't exist
    os.makedirs(raw_mask_dir, exist_ok=True)
    os.makedirs(colored_mask_dir, exist_ok=True)
    os.makedirs(overlay_dir, exist_ok=True)

    # Load the specified DeepLabV3 model
    model = create_deeplabv3_model(model_type)
    labels = load_labels(model)
    print('Model layers ', model)
    if quickone:
        exit(0)
        #just show model and exit

    #print('yyy ', model.state_dict().keys())
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if save_ckpt:
        ckpt={}
        ckpt['model']=model.state_dict()
        torch.save(ckpt, 'model_deeplabv3_pretrained.pt')
    if load_ckpt:
        ckpt = torch.load(ckptfn, map_location=device, weights_only=False)
        #print('xxx ckpt', ckpt)
        model.load_state_dict(ckpt['model'], strict=True)

    model.to(device)
    print(f"Using device: {device}")

    # Define the preprocessing transformations
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get the color palette
    palette = build_palette(19)
    print('xxx palette ', palette)
    #palette = get_color_palette()

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if not filenamex=="" and not filenamex==filename:
            continue
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            
            try:
                # Load the image
                input_image = Image.open(image_path).convert("RGB")
                original_size = input_image.size
                
                # Apply the transformations
                print('xxx ', original_size)
                input_tensor = preprocess(input_image)
                print('xxx input tensor shape', input_tensor.shape)
                input_batch = input_tensor.unsqueeze(0).to(device)
                print('xxx input batch shape', input_batch.shape)

                # Perform the forward pass without tracking gradients
                with torch.no_grad():
                    output = model(input_batch)['out']

                # Convert the output to a segmentation mask with class IDs
                output_predictions = output.argmax(1)[0]
               
                # Create and save the raw (grayscale) mask
                raw_mask_array = output_predictions.byte().cpu().numpy()
                uniq = np.unique(raw_mask_array).tolist()
                print('xxx uniq ', uniq)

                # FIX: Squeeze the array to remove the extra channel dimension
                raw_mask_array_squeezed = np.squeeze(raw_mask_array)

                raw_mask_image = Image.fromarray(raw_mask_array_squeezed).resize(original_size, Image.NEAREST)
                raw_mask_path = os.path.join(raw_mask_dir, os.path.splitext(filename)[0] + "_mask.png")

                raw_mask_image.save(raw_mask_path)
                
                # Create and save the colored mask
                colored_mask_image = Image.fromarray(raw_mask_array).resize(original_size, Image.NEAREST)
                colored_mask_image.putpalette(palette)
                colored_mask_path = os.path.join(colored_mask_dir, os.path.splitext(filename)[0] + "_color.png")
                colored_mask_image.save(colored_mask_path)
                
                print('xxx save  overlay')
                # Create and save the overlay image
                overlay_image = create_overlay(input_image, colored_mask_image)
                overlay_path = os.path.join(overlay_dir, os.path.splitext(filename)[0] + "_overlay.png")
                save_overlay(input_image, overlay_image,overlay_out=overlay_path, palette=palette, labels=labels, show_ids=uniq)
                # overlay_image.save(overlay_path)
                
                print(f"Processed '{filename}': Saved raw, colored, and overlay masks.")
            
            except Exception as e:
                print(f"Error processing image '{filename}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform semantic segmentation on a folder of images and save multiple output types.")
    parser.add_argument("--input_dir", type=str, default="data/apgdata/", help="Path to the directory containing input images.")
    parser.add_argument("--filename", type=str, default="", help="if empty string do all files")
    parser.add_argument("--output_dir", type=str, default="outputs/tmp/", help="Path to the directory containing input images.")
    parser.add_argument("--raw_mask_dir", type=str, default="masks", help="Path to the directory to save the raw grayscale masks.")
    parser.add_argument("--colored_mask_dir", type=str, default="color_mask", help="Path to the directory to save the colored masks.")
    parser.add_argument("--overlay_dir", type=str, default="overlay", help="Path to the directory to save the overlayed images.")
    parser.add_argument("--ckptfn", type=str, default="model_deeplabv3_pretrained.pt", help="ckpt filename")
    parser.add_argument('--load_ckpt', action='store_true')
    parser.add_argument('--quickone', action='store_true')
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet101"], 
                        help="DeepLabV3 backbone to use. Choose 'resnet50' or 'resnet101'.")

    args = parser.parse_args()
    raw_mask_dir = os.path.join(args.output_dir,args.raw_mask_dir)
    colored_mask_dir = os.path.join(args.output_dir, args.colored_mask_dir)
    overlay_dir = os.path.join(args.output_dir,args.overlay_dir)
    input_dir=args.input_dir+"images" 
    segment_and_save_images(input_dir, raw_mask_dir, colored_mask_dir, overlay_dir, args.model, args.load_ckpt, args.save_ckpt, args.ckptfn, filenamex=args.filename, quickone=args.quickone)

