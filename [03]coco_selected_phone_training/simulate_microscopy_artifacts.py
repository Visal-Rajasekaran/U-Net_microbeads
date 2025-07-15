'''import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import os
import glob


# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_JSON_PATH = os.path.join(BASE_DIR, "annotations.json")
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
CENTROIDS_FOLDER = os.path.join(BASE_DIR, "dataset", "centroids")
MASKS_FOLDER = os.path.join(BASE_DIR, "dataset", "masks")

# CONFIG

output_folder = os.path.join(BASE_DIR, "dataset","augmented_images")  # Output folder with same filenames
os.makedirs(output_folder, exist_ok=True)

def simulate_microscopy_artifacts(image):
    # Downsample & Upsample to simulate blur / lower resolution
    downscale_factor = 0.25
    small = cv2.resize(image, (0, 0), fx=downscale_factor, fy=downscale_factor, interpolation=cv2.INTER_AREA)
    image = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Add radial brightness falloff (vignetting)
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w / 2, h / 2
    radius = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    vignette = np.clip(1 - (radius / radius.max())**2, 0.4, 1)
    vignette = vignette[..., np.newaxis]
    image = (image * vignette).astype(np.uint8)

    # Simulate directional lighting gradient (brighter on left)
    gradient = np.tile(np.linspace(1.2, 0.6, w), (h, 1))
    image = np.clip(image.astype(np.float32) * gradient[..., None], 0, 255).astype(np.uint8)

    return image

# PROCESSING LOOP
image_paths = glob.glob("C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[[]03[]]coco_selected_phone_training\\dataset\\images\\*.png")


for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"Skipping unreadable image: {path}")
        continue

    aug = simulate_microscopy_artifacts(img)
    out_path = os.path.join(output_folder, os.path.basename(path))  # Preserve filename
    cv2.imwrite(out_path, aug)
    print(f"Processing: {path}")


print(f"Augmented images saved to: {output_folder}")'''


import cv2
import numpy as np
import os
import glob
import random

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "dataset", "augmented_images_v2")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def simulate_microscopy_artifacts(image):
    # Downsample & Upsample to simulate blur / lower resolution
    downscale_factor = 0.25
    small = cv2.resize(image, (0, 0), fx=downscale_factor, fy=downscale_factor, interpolation=cv2.INTER_AREA)
    image = cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    h, w = image.shape[:2]
    augmented = image.astype(np.float32)

    # Randomly apply vignetting (radial darkening)
    if random.random() < 0.5:
        Y, X = np.ogrid[:h, :w]
        center_x, center_y = w / 2, h / 2
        radius = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        vignette_strength = random.uniform(0.3, 0.6)
        vignette = np.clip(1 - (radius / radius.max())**2, vignette_strength, 1)
        vignette = vignette[..., np.newaxis]
        augmented *= vignette

    # Randomly apply directional lighting
    if random.random() < 0.5:
        direction = random.choice(["left", "right"])
        if direction == "left":
            gradient = np.tile(np.linspace(1.4, 0.4, w), (h, 1))
        else:  # right
            gradient = np.tile(np.linspace(0.4, 1.4, w), (h, 1))
        gradient = gradient[..., np.newaxis]
        augmented *= gradient

    return np.clip(augmented, 0, 255).astype(np.uint8)

# PROCESSING
image_paths = glob.glob("C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[[]03[]]coco_selected_phone_training\\dataset\\images\\*.png")
print(f"[INFO] Found {len(image_paths)} images in: {IMAGE_FOLDER}")

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Skipping unreadable image: {path}")
        continue

    aug = simulate_microscopy_artifacts(img)
    out_path = os.path.join(OUTPUT_FOLDER, os.path.basename(path))
    cv2.imwrite(out_path, aug)
    print(f"[OK] Processed: {os.path.basename(path)}")

print(f"[DONE] Augmented images saved to: {OUTPUT_FOLDER}")
