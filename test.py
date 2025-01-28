import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

# Replace 'net.Ushape_Trans' with the correct path to your model definition
from net.Ushape_Trans import Generator  

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
    min_depth = np.min(img_depth)
    max_depth = np.max(img_depth)
    # Avoid division by zero if min == max
    if max_depth != min_depth:
        img_depth = (img_depth - min_depth) / (max_depth - min_depth)
    return img_depth


def enhance_image(image):
    """
    Slightly reduce saturation and contrast compared to the original version.
    :param image: Input image as a torch tensor (C, H, W).
    :return: Enhanced image as a torch tensor.
    """
    # Convert from Torch tensor (C, H, W) to NumPy array (H, W, C) and denormalize
    image = image.detach().cpu().numpy().transpose(1, 2, 0)
    image = image * 255.0  # Scale to 0-255

    # Convert to HSV for saturation adjustment
    hsv_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2HSV)
    # Slightly reduce saturation (adjust this value to taste)
    hsv_image[..., 1] = cv2.add(hsv_image[..., 1], 10)  # Lower saturation increase
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

    # Apply a gentler contrast adjustment (alpha=1.05, beta=5)
    alpha = 1.05  # Contrast, reduced to avoid over-smoothing
    beta = 5      # Brightness, reduced to avoid excess brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Optional sharpening (adjust kernel to avoid excessive sharpness)
    kernel = np.array([[0, -0.5, 0],
                       [-0.5, 3, -0.5],
                       [0, -0.5, 0]])
    image = cv2.filter2D(image, -1, kernel)

    # Convert back to Torch tensor with values in [0, 1]
    image = torch.from_numpy(image.astype(np.float32) / 255.0).permute(2, 0, 1)
    return image


def process_image(image_path, depth_path, dtype, device):
    """
    Process input images and depth maps, ensuring proper resizing and normalization.
    """
    img_rgb = cv2.imread(image_path)
    if img_rgb is None:
        raise FileNotFoundError(f"RGB image not found: {image_path}")
    img_rgb = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_CUBIC)  # Higher quality resizing
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    img_rgb = img_rgb / 255.0  # Normalize to [0, 1]

    img_depth = cv2.imread(depth_path, 0)  # Load depth as grayscale
    if img_depth is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    img_depth = cv2.resize(img_depth, (256, 256), interpolation=cv2.INTER_CUBIC)  # Higher quality resizing
    img_depth = normalize_depth(img_depth)  # Normalize depth to [0, 1]
    img_depth = img_depth.reshape((256, 256, 1))

    # Combine RGB and depth into one tensor
    img_combined = np.concatenate((img_rgb, img_depth), axis=2)
    img_combined = torch.from_numpy(img_combined.astype(dtype)).permute(2, 0, 1).unsqueeze(0)

    return Variable(img_combined).to(device)


def load_model(generator_path, device):
    """
    Load the generator model with pretrained weights.
    """
    generator = Generator().to(device)
    generator_dict = generator.state_dict()

    # Load only matching layers from the pretrained file
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
    Example: '12_file.png' -> 12
    """
    parts = filename.split('_')[0]
    try:
        return int(parts)
    except ValueError:
        return None


def main():
    path_images = './dataset/UIEB/input'
    path_depth = './DPT/output_monodepth/UIEB'
    output_path = 'test/final'
    generator_path = 'save_model/generator_final.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = np.float32

    os.makedirs(output_path, exist_ok=True)
    generator = load_model(generator_path, device)

    # List all image files in your input folder
    image_files = [f for f in os.listdir(path_images) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    # Sort by numeric prefix if present
    image_files = sorted(image_files, key=lambda x: extract_numeric_prefix(x) or float('inf'))

    for i, img_file in enumerate(image_files):
        try:
            # Construct paths for the RGB and depth images
            img_path = os.path.join(path_images, img_file)
            depth_path = os.path.join(path_depth, f"{os.path.splitext(img_file)[0]}.png")

            # Load and process
            input_tensor = process_image(img_path, depth_path, dtype, device)
            output = generator(input_tensor)

            # Ensure output is a tensor and in the correct range
            output_tensor = output[3]  # Access the output tensor from the list (index 3 in this case)
            output_tensor = torch.clamp(output_tensor, 0, 1)

            # Enhance the final output
            enhanced_img = enhance_image(output_tensor.squeeze(0))  # Adjust as needed

            # Save the enhanced result (without extra normalization)
            save_image(enhanced_img, os.path.join(output_path, img_file), nrow=1, normalize=False)
            print(f"Processed {img_file} ({i + 1}/{len(image_files)})")

        except Exception as e:
            print(f"Error processing {img_file}: {e}")


if __name__ == "__main__":
    main()
