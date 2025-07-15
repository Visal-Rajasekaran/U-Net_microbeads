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

# Ensure we clean up memory whenever possible
gc.enable()
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_JSON_PATH = os.path.join(BASE_DIR, "annotations.json")
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images_new[4]")
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

class CentroidDataset(Dataset):
    def __init__(self, image_root, image_size=(512, 512), transform=None):
        self.image_root = image_root
        self.image_size = image_size
        self.transform = transform
        self.image_files = [
            fname for fname in os.listdir(image_root)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_root, filename)

        image = Image.open(image_path).convert("RGB").resize(self.image_size)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)  # shape: [3, H, W]

        return {
            'image': image,
            'filename': filename
        }

from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import copy
import matplotlib.pyplot as plt


def visualize_outputs(model_path, dataset, index=0, device='cuda'):

    # Load model
    model = NANO_PICS().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Get sample
    sample = dataset[index]
    image_tensor = sample['image'].unsqueeze(0).to(device)  # [1, 3, H, W]
    filename = sample['filename']

    with torch.no_grad():
        seg_pred, centroid_pred = model(image_tensor)

    # Convert predictions to numpy
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    seg_np = seg_pred.squeeze().cpu().numpy()
    centroid_np = centroid_pred.squeeze().cpu().numpy()

    # Threshold for segmentation mask
    #seg_binary = (seg_np >= 0.5).astype(np.uint8)

    # Count blobs from heatmap
    #num_peaks = count_local_maxima(centroid_np, threshold=0.1)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(image_np)
    axs[0].set_title("Input Image")
    axs[1].imshow(seg_np, cmap="gray")
    axs[1].set_title("Predicted Mask ")
    axs[2].imshow(centroid_np, cmap="hot")
    axs[2].set_title(f"Predicted Heatmap ")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"Augmented_model_v4_images_new_index_{index}.png")
    plt.close()

    print(f"Saved visualization for {filename} with  detected peaks.")


#print("Length of dataset", len(val_dataset))
model_path = os.path.join(BASE_DIR, "weights_phone_augmented_v3_model_v4.pth")
train_set = CentroidDataset(IMAGE_FOLDER)
for j in range(10):
    visualize_outputs(model_path, train_set, index=j)

