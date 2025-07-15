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
from loss_v5 import *
from model_v5 import *

# Ensure we clean up memory whenever possible
gc.enable()
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#COCO_JSON_PATH = os.path.join(BASE_DIR, "annotations.json")
IMAGE_FOLDER = os.path.join(BASE_DIR, "fft_1.0_selected")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
#CENTROIDS_FOLDER = os.path.join(BASE_DIR, "dataset", "centroids")
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

        count = torch.tensor([len(points)], dtype=torch.float32)
       
        return {
            'image': image,
            'seg_mask': mask,
            'centroid_map': heatmap,
            'count': count
        }

from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import copy

# CONFIG
batch_size = 4
learning_rate = 1e-4
val_split = 0.75
max_epochs = 500
early_stop_patience = 30
lr_plateau_patience = 14
generator = torch.Generator().manual_seed(42)

# Dataset
full_dataset = CentroidDataset(os.path.join(RESULTS_FOLDER, 'all_centroids.csv'), IMAGE_FOLDER, MASKS_FOLDER)
val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model and Loss
model = NANO_PICS().cuda()
criterion = MultiTaskLoss(dice_weight=1.0, bce_weight=1.0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_plateau_patience, verbose=True)

best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = float('inf')
epochs_no_improve = 0

# TRAINING LOOP
for epoch in range(max_epochs):
    model.train()
    running_train_loss = 0.0

    for batch in train_loader:
        images = batch['image'].cuda()
        seg_target = batch['seg_mask'].cuda()
        heatmap_target = batch['centroid_map'].cuda()
        count_target = batch['count'].cuda()

        #seg_pred, heat_pred, count_pred = model(images)
        #seg_pred, heat_pred = model(images)
        seg_pred = model(images)
        #print("Pred count:", count_pred.detach().cpu().numpy())
        #print("True count:", count_target.detach().cpu().numpy())


        #loss, (l_seg, l_heat, l_count) = criterion(seg_pred, seg_target, heat_pred, heatmap_target, count_pred, count_target)
        #loss, (l_seg, l_heat) = criterion(seg_pred, seg_target, heat_pred, heatmap_target)
        loss, l_seg = criterion(seg_pred, seg_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()
        print("OK")

    # Validation
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].cuda()
            seg_target = batch['seg_mask'].cuda()
            heatmap_target = batch['centroid_map'].cuda()
            count_target = batch['count'].cuda()

            #seg_pred, heat_pred, count_pred = model(images)
            #seg_pred, heat_pred = model(images)
            seg_pred = model(images)
            #val_loss, _ = criterion(seg_pred, seg_target, heat_pred, heatmap_target, count_pred, count_target)
            #val_loss, _ = criterion(seg_pred, seg_target, heat_pred, heatmap_target)
            val_loss, _ = criterion(seg_pred, seg_target)
            running_val_loss += val_loss.item()


    train_loss_avg = running_train_loss / len(train_loader)
    val_loss_avg = running_val_loss / len(val_loader)

    print(f"Epoch {epoch+1} | Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f}")

    scheduler.step(val_loss_avg)

    if val_loss_avg < best_val_loss:
        best_val_loss = val_loss_avg
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(best_model_wts, "weights_fft_new_microscopy_model_v5_0.75.pth")
        print("New best model saved.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= early_stop_patience:
            print(" Early stopping triggered.")
            break

print("Training complete.")

