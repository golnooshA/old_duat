import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from net.Ushape_Trans import Generator, Discriminator, weights_init_normal

# Paths
PATH_INPUT = './dataset/UIEB/input'
PATH_DEPTH = './DPT/output_monodepth/UIEB/'
PATH_GT = './dataset/UIEB/GT/'
# SAVE_DIR = './save_model/'
SAVE_DIR = '/content/drive/My Drive/My_Datasets/save_model/'

# Create the save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

class DepthDataset(Dataset):
    def __init__(self, input_path, depth_path, gt_path):
        self.input_list = sorted([f for f in os.listdir(input_path) if f.endswith('.png')])
        self.depth_list = sorted([f for f in os.listdir(depth_path) if f.endswith('.png')])
        self.gt_list = sorted([f for f in os.listdir(gt_path) if f.endswith('.png')])
        self.input_path = input_path
        self.depth_path = depth_path
        self.gt_path = gt_path

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input_file = self.input_list[idx]
        input_img = cv2.imread(os.path.join(self.input_path, input_file))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (256, 256))
        input_tensor = torch.from_numpy(input_img).permute(2, 0, 1).float() / 255.0

        depth_file = input_file.replace('.jpg', '.png')
        depth_img = cv2.imread(os.path.join(self.depth_path, depth_file), cv2.IMREAD_GRAYSCALE)
        depth_img = cv2.resize(depth_img, (256, 256))
        depth_tensor = torch.from_numpy(depth_img).unsqueeze(0).float() / 255.0

        # Combine RGB and Depth for 4-channel input
        real_A = torch.cat([input_tensor, depth_tensor.expand_as(input_tensor[:1])], dim=0)

        gt_img = cv2.imread(os.path.join(self.gt_path, input_file))
        gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)
        gt_img = cv2.resize(gt_img, (256, 256))
        gt_tensor = torch.from_numpy(gt_img).permute(2, 0, 1).float() / 255.0

        # Combine RGB and Depth for 4-channel output
        real_B = torch.cat([gt_tensor, depth_tensor.expand_as(gt_tensor[:1])], dim=0)

        return real_A, real_B

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and DataLoader
    train_dataset = DepthDataset(PATH_INPUT, PATH_DEPTH, PATH_GT)
    if len(train_dataset) == 0:
        raise ValueError("Dataset is empty. Please check your input paths.")
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
    print(f"Loaded dataset with {len(train_dataset)} valid samples.")

    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Loss functions and optimizers
    criterion_GAN = nn.MSELoss().to(device)
    criterion_pixelwise = nn.L1Loss().to(device)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0005, betas=(0.5, 0.999))
    
    n_epochs = 300
    save_freq = 5

    for epoch in range(n_epochs):
        print(f"Starting epoch {epoch+1}/{n_epochs}")
        for i, (real_A, real_B) in enumerate(train_loader):
            real_A, real_B = real_A.to(device), real_B.to(device)

            # Multi-scale inputs
            real_B_scales = [
                F.interpolate(real_B, size=(32, 32), mode="bilinear", align_corners=False),
                F.interpolate(real_B, size=(64, 64), mode="bilinear", align_corners=False),
                F.interpolate(real_B, size=(128, 128), mode="bilinear", align_corners=False),
                real_B,
            ]

            real_A_scales = [
                F.interpolate(real_A, size=(32, 32), mode="bilinear", align_corners=False),
                F.interpolate(real_A, size=(64, 64), mode="bilinear", align_corners=False),
                F.interpolate(real_A, size=(128, 128), mode="bilinear", align_corners=False),
                real_A,
            ]

            # Generator forward
            fake_B = generator(real_A)
            fake_B_scales = [
                F.interpolate(fake_B[-1], size=(32, 32), mode="bilinear", align_corners=False),
                F.interpolate(fake_B[-1], size=(64, 64), mode="bilinear", align_corners=False),
                F.interpolate(fake_B[-1], size=(128, 128), mode="bilinear", align_corners=False),
                fake_B[-1],
            ]

            # Discriminator loss
            pred_fake = discriminator(fake_B_scales, real_A_scales)
            pred_real = discriminator(real_B_scales, real_A_scales)

            loss_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            loss_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = 0.5 * (loss_real + loss_fake)

            optimizer_D.zero_grad()
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            # Generator loss
            pred_fake = discriminator(fake_B_scales, real_A_scales)
            loss_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_pixel = criterion_pixelwise(fake_B[-1], real_B)
            loss_G = loss_GAN + 100 * loss_pixel

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            print(f"[Epoch {epoch+1}/{n_epochs}] [Batch {i+1}/{len(train_loader)}] [D loss: {loss_D.item():.4f}] [G loss: {loss_G.item():.4f}]")

        # Save models and optimizers every 5 epochs
        if (epoch + 1) % save_freq == 0:
            generator_path = os.path.join(SAVE_DIR, f'generator_epoch_{epoch+1}.pth')
            discriminator_path = os.path.join(SAVE_DIR, f'discriminator_epoch_{epoch+1}.pth')
            optimizers_path = os.path.join(SAVE_DIR, f'optimizers_epoch_{epoch+1}.pth')

            torch.save(generator.state_dict(), generator_path)
            torch.save(discriminator.state_dict(), discriminator_path)
            torch.save({
                'optimizer_G': optimizer_G.state_dict(),
                'optimizer_D': optimizer_D.state_dict(),
            }, optimizers_path)

            print(f"Saved models and optimizers for epoch {epoch+1} at: {SAVE_DIR}")

    # Save final models
    torch.save(generator.state_dict(), os.path.join(SAVE_DIR, 'generator_final.pth'))
    torch.save(discriminator.state_dict(), os.path.join(SAVE_DIR, 'discriminator_final.pth'))
    torch.save({
        'optimizer_G': optimizer_G.state_dict(),
        'optimizer_D': optimizer_D.state_dict(),
    }, os.path.join(SAVE_DIR, 'optimizers_final.pth'))
    print("Training complete. Models saved.")
