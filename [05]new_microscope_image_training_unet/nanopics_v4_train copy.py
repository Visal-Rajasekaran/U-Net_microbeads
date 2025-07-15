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
from loss import *
from model_v4 import *
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
# Ensure we clean up memory whenever possible
gc.enable()
print(torch.cuda.is_available())

# Define paths
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
COCO_JSON_PATH = os.path.join(BASE_DIR, "annotations.json")
IMAGE_FOLDER = "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\dataset\\grayscale_norm_selected_split\\more_than_300"
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
CENTROIDS_FOLDER = os.path.join(BASE_DIR, "dataset", "centroids")
MASKS_FOLDER = "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\dataset\\grayscale_norm_selected_masks_dense"

def generate_heatmap(image_shape, points, radii, default_sigma=4, clip_val=1.0, threshold=0.01):
    heatmap = np.zeros(image_shape, dtype=np.float32)
    for (x, y), radius in zip(points, radii):
        sigma = radius / 3 if radius else default_sigma
        tmp = np.zeros(image_shape, dtype=np.float32)
        cx, cy = round(x), round(y)
        if 0 <= cy < image_shape[0] and 0 <= cx < image_shape[1]:
            tmp[cy, cx] = 1.0
            tmp = gaussian_filter(tmp, sigma=sigma, mode='constant')
            #tmp = np.where(tmp > threshold, tmp, 0)
            tmp/=tmp.max()
            heatmap += tmp
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
            'count': count,
            'points':points
        }

from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import copy

# CONFIG
batch_size = 4
learning_rate = 1e-4
val_split = 0.2
max_epochs = 200
early_stop_patience = 30
lr_plateau_patience = 14
generator = torch.Generator().manual_seed(42)

# Dataset
full_dataset = CentroidDataset(os.path.join(IMAGE_FOLDER, 'dense_centroids.csv'), IMAGE_FOLDER, MASKS_FOLDER)
val_size = int(len(full_dataset) * val_split)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],generator=generator)

def visualize_outputs(model_path, dataset, index=0, device='cuda', type=None):
    """
    Loads the trained model and visualizes image, predicted segmentation, and predicted centroid heatmap.
    Overlays predicted centroids on both the input image and predicted heatmap.
    """
    # Load model
    model = NANO_PICS().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
    model.eval()

    # Load one sample
    sample = dataset[index]
    image = sample['image'].unsqueeze(0).to(device)  # [1, 3, H, W]
    gt_counts = sample['count']

    with torch.no_grad():
        seg_pred, centroid_pred = model(image)

    # Detach + convert to numpy
    image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
    seg_np = seg_pred.squeeze().cpu().numpy()
    centroid_np = centroid_pred.squeeze().cpu().numpy()
    seg_target = sample['seg_mask'].squeeze().numpy()
    centroid_target = sample['centroid_map'].squeeze().numpy()

    thresholded_seg_np = (seg_np >= 0.9).astype(np.uint8)

   
    input_gt = (image_np * 255).astype(np.uint8).copy()
    input_gt = cv2.cvtColor(input_gt, cv2.COLOR_RGB2BGR)
    for (x, y) in sample['points']:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(input_gt, (cx, cy), 5, (0, 0, 255), 1)  # blue for GT

    # -- Predicted Centroids --
    pred_binary = (centroid_np >= 0.3).astype(np.uint8)
    pred_num_labels, _, _, pred_centroids = cv2.connectedComponentsWithStats(pred_binary)

    annotated_input = (image_np * 255).astype(np.uint8).copy()
    annotated_input = cv2.cvtColor(annotated_input, cv2.COLOR_RGB2BGR)
    for i in range(1, pred_num_labels):
        x, y = int(pred_centroids[i][0]), int(pred_centroids[i][1])
        cv2.circle(annotated_input, (x, y), 5, (0, 255, 0), 1)  # green for prediction

    heatmap_overlay = cv2.cvtColor((centroid_np * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    for i in range(1, pred_num_labels):
        x, y = int(pred_centroids[i][0]), int(pred_centroids[i][1])
        cv2.circle(heatmap_overlay, (x, y), 5, (0, 255, 0), 1)

    # -- Plot --
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(cv2.cvtColor(input_gt, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title(f"Input with GT Centroids (GT: {int(gt_counts[0])})")

    axs[0, 1].imshow(seg_target, cmap='gray')
    axs[0, 1].set_title("Ground Truth Mask")

    axs[0, 2].imshow(centroid_target, cmap='gray')
    axs[0, 2].set_title("Ground Truth Heatmap")

    axs[1, 0].imshow(cv2.cvtColor(annotated_input, cv2.COLOR_BGR2RGB))
    axs[1, 0].set_title(f"Input with Predicted Centroids (Pred: {pred_num_labels - 1}, GT: {int(gt_counts[0])})")

    axs[1, 1].imshow(seg_np, cmap='gray')
    axs[1, 1].set_title("Predicted Mask")

    axs[1, 2].imshow(cv2.cvtColor(heatmap_overlay, cv2.COLOR_BGR2RGB))
    axs[1, 2].set_title(f"Pred Heatmap + Centroids (Pred: {pred_num_labels - 1})")

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    model_folder = "grayscale_norm_dense_model_v4_0.2"
    os.makedirs(os.path.join(RESULTS_FOLDER, model_folder), exist_ok=True)
    plt.savefig(os.path.join(RESULTS_FOLDER, model_folder, f"Model_v4_fft_{type}_{index}.png"))
    plt.close()

print("Length of dataset", len(val_dataset))
model_path = os.path.join(BASE_DIR,"weights" ,"weights_grayscale_norm_dense_model_v4_0.2.pth")
#train_set = CentroidDataset(os.path.join(RESULTS_FOLDER, 'all_centroids_copy.csv'), IMAGE_FOLDER, MASKS_FOLDER)
for j in range(len(val_dataset)):
    visualize_outputs(model_path, val_dataset, index=j,type="val")
for j in range(len(train_dataset)):
    visualize_outputs(model_path, train_dataset, index=j,type='train')
print("DONE")
