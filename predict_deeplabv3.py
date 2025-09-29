import torch
from torchvision import models, transforms
from PIL import Image
import os
import argparse
import numpy as np
# load stock pretrained and dump to local file deepv3 resnet50

def create_deeplabv3_model(model_type: str):
    """Loads a pretrained DeepLabV3 model with the specified backbone."""
    if model_type == 'resnet101':
        return models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT).eval()
    elif model_type == 'resnet50':
        return models.segmentation.deeplabv3_resnet50(weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT).eval()
    else:
        raise ValueError("Invalid model_type. Choose 'resnet50' or 'resnet101'.")

def get_color_palette():
    """Returns a color palette for the 21 classes (including background)."""
    palette = torch.tensor([2**25 - 1, 2**15 - 1, 2**21 - 1])
    colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
    return (colors % 255).numpy().astype("uint8")

def create_overlay(original_image, colored_mask, alpha=0.5):
    """Blends the original image and the colored mask."""
    overlay = Image.blend(original_image.convert("RGB"), colored_mask.convert("RGB"), alpha)
    return overlay

def segment_and_save_images(input_dir: str, raw_mask_dir: str, colored_mask_dir: str, overlay_dir: str, model_type: str = 'resnet50', load_ckpt=False, save_ckpt=False, ckptfn="model_deeplabv3_pretrained.pt"):
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
    print('xxx ', model)
    print('yyy ', model.state_dict().keys())
    #exit(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if save_ckpt:
        torch.save(model.state_dict(), 'model_deeplabv3_pretrained.pt')
    if load_ckpt:
        ckpt = torch.load(ckptfn, map_location=device, weights_only=False)
        print('xxx ckpt', ckpt)
        model.load_state_dict(ckpt['model'], strict=True)

    model.to(device)
    print(f"Using device: {device}")

    # Define the preprocessing transformations
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Get the color palette
    palette = get_color_palette()

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
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
                print('xxx save mask', output_predictions.shape)
               
                # Create and save the raw (grayscale) mask
                raw_mask_array = output_predictions.byte().cpu().numpy()

                # FIX: Squeeze the array to remove the extra channel dimension
                raw_mask_array_squeezed = np.squeeze(raw_mask_array)

                raw_mask_image = Image.fromarray(raw_mask_array_squeezed).resize(original_size, Image.NEAREST)
                print('xxx save mask2')
                raw_mask_path = os.path.join(raw_mask_dir, os.path.splitext(filename)[0] + "_raw.png")

                print('xxx save mask2')
                raw_mask_image.save(raw_mask_path)

                print('xxx save colored mask')
                
                # Create and save the colored mask
                colored_mask_image = Image.fromarray(raw_mask_array).resize(original_size, Image.NEAREST)
                colored_mask_image.putpalette(palette)
                colored_mask_path = os.path.join(colored_mask_dir, os.path.splitext(filename)[0] + "_color.png")
                colored_mask_image.save(colored_mask_path)
                
                print('xxx save  overlay')
                # Create and save the overlay image
                overlay_image = create_overlay(input_image, colored_mask_image)
                overlay_path = os.path.join(overlay_dir, os.path.splitext(filename)[0] + "_overlay.png")
                overlay_image.save(overlay_path)
                
                print(f"Processed '{filename}': Saved raw, colored, and overlay masks.")
            
            except Exception as e:
                print(f"Error processing image '{filename}': {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform semantic segmentation on a folder of images and save multiple output types.")
    parser.add_argument("--input_dir", type=str, default="data/apgdata/", help="Path to the directory containing input images.")
    parser.add_argument("--output_dir", type=str, default="outputs/tmp/", help="Path to the directory containing input images.")
    parser.add_argument("--raw_mask_dir", type=str, default="mask", help="Path to the directory to save the raw grayscale masks.")
    parser.add_argument("--colored_mask_dir", type=str, default="color_mask", help="Path to the directory to save the colored masks.")
    parser.add_argument("--overlay_dir", type=str, default="overlay", help="Path to the directory to save the overlayed images.")
    parser.add_argument("--ckptfn", type=str, default="model_deeplabv3_pretrained.pt", help="ckpt filename")
    parser.add_argument('--load_ckpt', action='store_true')
    parser.add_argument('--save_ckpt', action='store_true')
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "resnet101"], 
                        help="DeepLabV3 backbone to use. Choose 'resnet50' or 'resnet101'.")

    args = parser.parse_args()
    raw_mask_dir = args.output_dir+args.raw_mask_dir
    colored_mask_dir = args.output_dir+args.colored_mask_dir
    overlay_dir = args.output_dir+args.overlay_dir
    input_dir=args.input_dir+"images" 
    segment_and_save_images(input_dir, raw_mask_dir, colored_mask_dir, overlay_dir, args.model, args.load_ckpt, args.save_ckpt, args.ckptfn)

