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
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
COCO_JSON_PATH = os.path.join(BASE_DIR,"label_generation", "annotations.json")
IMAGE_FOLDER = os.path.join(BASE_DIR,"dataset", "fft_filtered")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
#CENTROIDS_FOLDER = os.path.join(BASE_DIR, "dataset", "centroids")
#MASKS_FOLDER = os.path.join(BASE_DIR, "dataset", "masks")

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
import matplotlib.pyplot as plt
# CONFIG
batch_size = 4
learning_rate = 1e-4
val_split = 0.2
max_epochs = 200
early_stop_patience = 30
lr_plateau_patience = 14

# Dataset
full_dataset = CentroidDataset(os.path.join(RESULTS_FOLDER, 'all_centroids.csv'), IMAGE_FOLDER, MASKS_FOLDER)
val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model and Loss
model = NANO_PICS().cuda()
criterion = MultiTaskLoss(dice_weight=2.0, bce_weight=1.0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=lr_plateau_patience, verbose=True)

best_model_wts = copy.deepcopy(model.state_dict())
best_val_loss = float('inf')
epochs_no_improve = 0

def visualize_outputs(model_path, dataset, index=0, device='cuda',type=None):
    """
    Loads the trained model and visualizes image, predicted segmentation, and predicted centroid heatmap.

    Args:
        model_path (str): Path to the saved model state_dict (.pth file)
        dataset (Dataset): An instance of CentroidDataset
        index (int): Index of the sample to visualize
        device (str): 'cuda' or 'cpu'
    """
    # Load model
    model = NANO_PICS().cuda()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Load one sample
    sample = dataset[index]
    image = sample['image'].unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        #seg_pred, centroid_pred = model(image)
        seg_pred = model(image)

    # Detach + convert to numpy
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    seg_np = seg_pred.squeeze().cpu().numpy()
    #centroid_np = centroid_pred.squeeze().cpu().numpy()
    seg_target = sample['seg_mask'].squeeze().numpy()
    centroid_target = sample['centroid_map'].squeeze().numpy()

    #Thresholding
    thresholded_seg_np = (seg_np >= 0.9).astype(np.uint8)

    # Plot
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(image_np)
    axs[0, 0].set_title("Input Image")
    axs[0, 1].imshow(seg_target, cmap='gray')
    axs[0, 1].set_title("Ground Truth Mask")
    axs[0, 2].imshow(centroid_target, cmap='gray')
    axs[0, 2].set_title("Ground Truth Heatmap")

    axs[1, 0].imshow(image_np)
    axs[1, 0].set_title("Input Image (Again)")
    axs[1, 1].imshow(seg_np, cmap='gray')
    axs[1, 1].set_title("Predicted Mask")
    axs[1, 2].imshow(seg_np, cmap='gray')
    axs[1, 2].set_title("Predicted Centroid Heatmap")

    # Fit GMM to predicted heatmap
    #means, covariances = fit_gaussians_to_heatmap(centroid_np, n_components=400)

    #means, covariances = estimate_best_gmm_components(thresholded_seg_np, max_components=500)

    '''if means is not None:
        axs[1, 2].scatter(means[:, 1], means[:, 0], s=50, c='cyan', marker='x', label='GMM Peaks')
        axs[1, 2].legend(loc='lower right')'''

    
    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    model_folder = "fft_model_v4_seg_only_0.75"
    os.makedirs(os.path.join(RESULTS_FOLDER,model_folder),exist_ok=True)
    plt.savefig(os.path.join(RESULTS_FOLDER,model_folder,f"Model_v4_fft_{type}_{index}.png"))

    '''markers, bead_count, overlay = apply_watershed(thresholded_seg_np, original_img=image_np)
    print("Bead count (watershed):", bead_count)
    plt.imsave("watershed_overlay.png", overlay)'''

print("Length of dataset", len(val_dataset))
model_path = os.path.join(BASE_DIR,"weights" "weights_fft_new_microscopy_model_v5_seg_only_0.75.pth")
#train_set = CentroidDataset(os.path.join(RESULTS_FOLDER, 'all_centroids_copy.csv'), IMAGE_FOLDER, MASKS_FOLDER)
for j in range(len(val_dataset)):
    visualize_outputs(model_path, val_dataset, index=j,type="val")
for j in range(len(train_dataset)):
    visualize_outputs(model_path, train_dataset, index=j,type='train')

