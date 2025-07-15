from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gc
import os
from datetime import datetime

# Ensure we clean up memory whenever possible
gc.enable()
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_JSON_PATH = os.path.join(BASE_DIR, "annotations.json")
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "augmented_images_directional_only")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
CENTROIDS_FOLDER = os.path.join(BASE_DIR, "dataset", "centroids")
MASKS_FOLDER = os.path.join(BASE_DIR, "dataset", "masks")

def generate_heatmap(image_shape, points, radii, default_sigma=4):
    heatmap = np.zeros(image_shape, dtype=np.float32)
    for (x, y), radius in zip(points, radii):
        sigma = radius / 3 if radius else default_sigma #radius/2 seems better than typical /3
        tmp = np.zeros(image_shape, dtype=np.float32)
        tmp[int(y), int(x)] = 1
        tmp = cv2.GaussianBlur(tmp, (0,0), sigma)
        tmp /= tmp.max()
        heatmap = np.maximum(heatmap, tmp)
    return heatmap



import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
'''class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)'''

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        u4 = self.up4(b)
        a4 = self.att4(g=u4, x=s4)
        d4 = self.dec4(torch.cat([u4, a4], dim=1))

        u3 = self.up3(d4)
        a3 = self.att3(g=u3, x=s3)
        d3 = self.dec3(torch.cat([u3, a3], dim=1))

        u2 = self.up2(d3)
        a2 = self.att2(g=u2, x=s2)
        d2 = self.dec2(torch.cat([u2, a2], dim=1))

        u1 = self.up1(d2)
        a1 = self.att1(g=u1, x=s1)
        d1 = self.dec1(torch.cat([u1, a1], dim=1))

        return torch.sigmoid(self.out(d1))

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return torch.sigmoid(self.out(d1))
    
class NANO_PICS(nn.Module):
    def __init__(self):
        super().__init__()
        self.segmentation_net = UNet(in_channels=3, out_channels=1)
        self.centroid_net = UNet(in_channels=3, out_channels=1)

    def forward(self, x):
        seg_out = self.segmentation_net(x)
        centroid_out = self.centroid_net(x)
        return seg_out, centroid_out

def bce_loss(pred, target):
    return F.binary_cross_entropy(pred, target)

def mse_loss(pred, target):
    return F.mse_loss(pred, target)

def combined_loss(seg_pred, seg_target, centroid_pred, centroid_target):
    l_seg = F.binary_cross_entropy(seg_pred, seg_target)
    l_centroid = F.binary_cross_entropy(centroid_pred, centroid_target)
    return l_seg + l_centroid

class CentroidDataset(Dataset):
    def __init__(self, csv_path, image_root, mask_root, image_size=(512, 512), sigma=4, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.mask_root = mask_root
        self.image_size = image_size
        self.sigma = sigma
        self.transform = transform
        self.grouped = self.df.groupby("image_id")

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, idx):
        image_id = list(self.grouped.groups.keys())[idx]
        group = self.grouped.get_group(image_id)

        filename = group["image_filename"].values[0]
        base_name = os.path.splitext(filename)[0]

        # Load RGB image
        image_path = os.path.join(self.image_root, filename)
        image = Image.open(image_path).convert("RGB").resize(self.image_size)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)  # shape: [3, H, W]

        # Load segmentation mask (grayscale) and convert to binary tensor
        mask_path = os.path.join(self.mask_root, base_name + ".png")
        mask = Image.open(mask_path).convert("L").resize(self.image_size)
        mask = T.ToTensor()(mask)       # shape: [1, H, W], float in [0,1]

        # Threshold the mask if needed (some images might be soft gray)
        mask = mask.float()

        # Load centroids
        points = list(zip(group['x'], group['y']))
        radii = [row["approx_radius"] for _, row in group.iterrows()]
        heatmap = generate_heatmap(self.image_size, points, radii, default_sigma=self.sigma)
        heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)  # [1, H, W]

        return {
            'image': image,
            'seg_mask': mask,
            'centroid_map': heatmap
        }


from torch.utils.data import DataLoader
import torch.optim as optim

train_set = CentroidDataset(os.path.join(RESULTS_FOLDER, 'all_centroids.csv'), IMAGE_FOLDER, MASKS_FOLDER)
train_loader = DataLoader(train_set, batch_size=2, shuffle=True)

model = NANO_PICS().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
print("Starting...")
for epoch in range(200):
    for i,batch in enumerate(train_loader):
        images = batch['image'].cuda()
        seg_target = batch['seg_mask'].cuda()
        centroid_target = batch['centroid_map'].cuda()

        seg_pred, centroid_pred = model(images)

        loss = combined_loss(seg_pred, seg_target, centroid_pred, centroid_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {loss.item():.4f}")

        # Garbage collection every 10 batches
        if i % 10 == 0:
            del images, seg_target, centroid_target, seg_pred, centroid_pred, loss
            torch.cuda.empty_cache()
            gc.collect()



torch.save(model.state_dict(), "weights_phone_directional.pth")


