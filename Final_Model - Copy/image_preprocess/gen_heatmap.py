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
#from loss_v5 import *
#from model_v5 import *

# Ensure we clean up memory whenever possible
'''gc.enable()
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))
'''
# Define paths
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)
COCO_JSON_PATH = os.path.join(BASE_DIR, "annotations.json")
IMAGE_FOLDER = "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\dataset\\fft_0.9_selected"
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
CENTROIDS_FOLDER = "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\all_centroids.csv"
MASKS_FOLDER = "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\dataset\\masks"

from scipy.ndimage import gaussian_filter

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
            'count': count
        }

from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import copy
import matplotlib.pyplot as plt



# Dataset
full_dataset = CentroidDataset(CENTROIDS_FOLDER, IMAGE_FOLDER, MASKS_FOLDER)

for idx, sample in enumerate(full_dataset):
    heatmap = sample['centroid_map'].squeeze(0).numpy()
    plt.imsave(f"Image{idx}.png", heatmap, cmap='gray')
    plt.close()
