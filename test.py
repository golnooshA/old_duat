import os
import torch
import numpy as np
import cv2
from torch.autograd import Variable
from torchvision.utils import save_image
import torch.nn.functional as F
from net.Ushape_Trans import Generator  # Ensure correct path to your model

# Ensure no multiprocessing conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def split(img):
    output = []
    output.append(F.interpolate(img, scale_factor=0.125))
    output.append(F.interpolate(img, scale_factor=0.25))
    output.append(F.interpolate(img, scale_factor=0.5))
    output.append(img)
    return output

def normalize_depth(img_depth):
    """
    Normalize depth map to the range [0, 1].
    """
    img_depth = img_depth.astype(np.float32)
    return (img_depth - np.min(img_depth)) / (np.max(img_depth) - np.min(img_depth))

def process_image(image_path, depth_path, dtype, device):
    """
    Process input images and depth maps, ensuring proper resizing and normalization.
    """
    img_rgb = cv2.imread(image_path)
    if img_rgb is None:
        raise FileNotFoundError(f"RGB image not found: {image_path}")
    img_rgb = cv2.resize(img_rgb, (256, 256))
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb / 255.0  # Normalize RGB to [0, 1]

    img_depth = cv2.imread(depth_path, 0)  # Load depth as grayscale
    if img_depth is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    img_depth = cv2.resize(img_depth, (256, 256))
    img_depth = normalize_depth(img_depth)  # Normalize depth to [0, 1]
    img_depth = img_depth.reshape((256, 256, 1))

    img_combined = np.concatenate((img_rgb, img_depth), axis=2)  # Combine RGB and Depth
    img_combined = torch.from_numpy(img_combined.astype(dtype)).permute(2, 0, 1).unsqueeze(0)
    return Variable(img_combined).to(device)

def adjust_dynamic_brightness_contrast(img, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for dynamic contrast enhancement.
    """
    img_np = img.cpu().detach().numpy().squeeze().transpose(1, 2, 0)
    img_np = cv2.cvtColor((img_np * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    img_np[:, :, 0] = clahe.apply(img_np[:, :, 0])  # Apply CLAHE on the L-channel
    img_np = cv2.cvtColor(img_np, cv2.COLOR_LAB2RGB)
    img_np = img_np.astype(np.float32) / 255.0
    return torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(img.device)

def load_model(generator_path, device):
    """
    Load the generator model with pretrained weights.
    """
    generator = Generator().to(device)
    generator_dict = generator.state_dict()
    
    # Load pretrained weights selectively
    pretrained_dict = torch.load(generator_path, map_location=device)
    filtered_dict = {
        k: v for k, v in pretrained_dict.items() if k in generator_dict and v.size() == generator_dict[k].size()
    }
    generator_dict.update(filtered_dict)
    generator.load_state_dict(generator_dict)
    
    generator.eval()
    return generator

def extract_numeric_prefix(filename):
    """
    Extract numeric prefix from filenames for proper sorting.
    """
    parts = filename.split('_')[0]
    try:
        return int(parts)
    except ValueError:
        return None

def main():
    path_images = './dataset/UIEB/input'
    path_depth = './DPT/output_monodepth/UIEB'
    output_path = 'test/new'
    generator_path = 'save_model/generator_epoch_5.pth'
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = np.float32

    os.makedirs(output_path, exist_ok=True)

    generator = load_model(generator_path, device)

    image_files = [f for f in os.listdir(path_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files = sorted(image_files, key=lambda x: extract_numeric_prefix(x) or float('inf'))

    for i, img_file in enumerate(image_files):
        try:
            img_path = os.path.join(path_images, img_file)
            depth_path = os.path.join(path_depth, f"{os.path.splitext(img_file)[0]}.png")  # Ensure correct depth path

            input_tensor = process_image(img_path, depth_path, dtype, device)

            output = generator(input_tensor)
            
            # Dynamically adjust brightness and contrast of the enhanced image
            enhanced_img = adjust_dynamic_brightness_contrast(output[3].data)

            # Save the enhanced image
            save_image(enhanced_img, os.path.join(output_path, img_file), nrow=1, normalize=True)
            print(f"Processed {img_file} ({i + 1}/{len(image_files)})")
        except Exception as e:
            print(f"Error processing {img_file}: {e}")

if __name__ == "__main__":
    main()
