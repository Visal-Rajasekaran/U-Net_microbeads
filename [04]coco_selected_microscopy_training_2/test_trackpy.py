import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
#print(os.getcwd())
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import trackpy as tp
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_JSON_PATH = os.path.join(BASE_DIR, "annotations.json")
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
CENTROIDS_FOLDER = os.path.join(BASE_DIR, "dataset", "centroids")
MASKS_FOLDER = os.path.join(BASE_DIR, "dataset", "masks")


img_path = os.path.join(IMAGE_FOLDER, "3_320x stock m270_4xobj_3_grid_r0_c6.png")
img = cv2.imread(img_path)  # BGR by default

if img is None:
    raise ValueError(f"Image not found at {img_path}")

# Convert to grayscale for Trackpy
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Optional: convert to float for subpixel accuracy
gray_float = gray.astype(np.float32)

# --- Locate particles ---
# diameter: approx particle size (must be odd), adjust to match bead size
# minmass: lower this if it's not detecting enough features
f = tp.locate(gray_float, diameter=3, minmass=5)

# Print some of the results
print(f.head())

# --- Plot results ---
fig, ax = plt.subplots(figsize=(8, 8))
tp.annotate(f, gray, ax=ax)  # you can also use `gray_float` here
plt.show()