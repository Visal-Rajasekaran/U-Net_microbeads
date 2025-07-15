import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
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
from loss import *
from model_v4 import *


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\dataset\\fft_1.0_selected_80x"
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")



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

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(image_np)
    axs[0].set_title("Input Image")
    axs[1].imshow(seg_np, cmap="gray")
    axs[1].set_title("Predicted Mask ")
    axs[2].imshow(centroid_np, cmap="gray")
    axs[2].set_title(f"Predicted Heatmap")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"Test_1_Balanced{filename}.png")
    plt.close()

    print(f"Saved visualization for {filename} ")


model_path = os.path.join(BASE_DIR, "weights_fft_0.9_new_microscopy_model_v4_0.5.pth")
train_set = CentroidDataset(IMAGE_FOLDER)

for i in range(8):
    visualize_outputs(model_path, train_set, index=i)


