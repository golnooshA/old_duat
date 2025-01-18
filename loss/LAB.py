import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utility.ptcolor import rgb2lab  # Ensure this module exists and is in your path
from utility.Qnt import quantAB
import torch
import torch.nn.functional as F
import torch.nn as nn


class lab_Loss(nn.Module):
    def __init__(self, alpha=1, weight=1, levels=7, vmin=-80, vmax=80):
        super(lab_Loss, self).__init__()
        self.alpha = alpha
        self.weight = weight
        self.levels = levels
        self.vmin = vmin
        self.vmax = vmax

    def Hist_2_Dist_L(self, img, tab, alpha):
        img_dist = ((img.unsqueeze(1) - tab) ** 2)
        p = F.softmax(-alpha * img_dist, dim=1)
        return p

    def Hist_2_Dist_AB(self, img, tab, alpha):
        img_dist = ((img.unsqueeze(1) - tab) ** 2).sum(2)
        p = F.softmax(-alpha * img_dist, dim=1)
        return p

    def loss_ab(self, img, gt, alpha, tab, levels):
        p = self.Hist_2_Dist_AB(img, tab, alpha).to(img.device)
        q = self.Hist_2_Dist_AB(gt, tab, alpha).to(gt.device)
        p = torch.clamp(p, 0.001, 0.999)
        loss = -(q * torch.log(p)).sum([1, 2, 3]).mean()
        return loss

    def forward(self, img, gt):
        tab = quantAB(self.levels, self.vmin, self.vmax).to(img.device)
        lab_img = torch.clamp(rgb2lab(img), self.vmin, self.vmax)
        lab_gt = torch.clamp(rgb2lab(gt), self.vmin, self.vmax)

        # Compute L channel loss
        loss_l = torch.abs(lab_img[:, 0, :, :] - lab_gt[:, 0, :, :]).mean()
        
        # Compute AB channel loss
        loss_AB = self.loss_ab(lab_img[:, 1:, :, :], lab_gt[:, 1:, :, :], self.alpha, tab, self.levels)
        
        # Combine losses
        loss = loss_l + self.weight * loss_AB
        return loss


if __name__ == "__main__":
    # Simple test to check the implementation
    img = torch.randn(1, 3, 256, 256).cuda()  # Replace with actual test images
    gt = torch.randn(1, 3, 256, 256).cuda()  # Replace with actual ground truth images

    lab_loss = lab_Loss().cuda()
    loss = lab_loss(img, gt)
    print("LAB Loss:", loss)
