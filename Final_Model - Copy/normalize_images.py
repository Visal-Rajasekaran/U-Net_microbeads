import numpy as np
import cv2
import os
from pathlib import Path
from skimage import exposure
import shutil
import random
import csv
from datetime import datetime
import matplotlib.pyplot as plt


BASE_FOLDER = Path(__file__).parent
INPUT_ROOT = BASE_FOLDER / "dataset2" / "images_selected"
OUTPUT_FOLDER = BASE_FOLDER / "dataset2" / "images_selected_gray"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# -------------------- Bead Detection -------------------- #
def normalize_image_rgb(image, method='none'):
    if method == 'mean_std':
        norm = image.astype(np.float32)
        mean = norm.mean()
        std = norm.std()
        if std < 1e-5: std = 1
        norm = (norm - mean) / std * 40 + 128
        return np.clip(norm, 0, 255).astype(np.uint8)

    elif method == 'clahe':
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)
        '''# Convert to LAB and apply CLAHE on L channel
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)'''

    elif method == 'histogram':
        norm = image.astype(np.float32)
        imin, imax = np.percentile(norm, (2, 98))
        if imax - imin < 1e-5: return image
        norm = (norm - imin) / (imax - imin) * 255
        return np.clip(norm, 0, 255).astype(np.uint8)
    elif method == 'min_max':    
        norm_image = np.zeros_like(image, dtype=np.float32)
        '''for c in range(3):  # For each color channela
            channel = image[:, :, c].astype(np.float32)
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:  # Avoid division by zero
                norm_image[:, :, c] = (channel - min_val) / (max_val - min_val)
            else:
                norm_image[:, :, c] = 0  # flat color'''

        channel = image.astype(np.float32)
        min_val = channel.min()
        max_val = channel.max()
        if max_val > min_val:  # Avoid division by zero
            norm_image = (channel - min_val) / (max_val - min_val)
        else:
            norm_image = 0  # flat color
        return (norm_image * 255).astype(np.uint8)

    else:
        return image  # No normalization



# -------------------- Main -------------------- #
if __name__ == "__main__":
    print("Started")

    density_config = {
        "80x": (1000, (1, 15), 7),
        "320x": (500, (1, 4), 8),
        "640x": (300, (1, 2), 9),
        "1280x": (100, (1, 2), 10),
        "2560x": (50, (1, 2), 10),
        "5120x": (20, (1, 2), 10),
        "10240x": (10, (1, 1), 10),
    }

    img_count = 0
    for density, (num_clusters, cluster_range,_) in density_config.items():
        density_path = INPUT_ROOT / density
        print(density_path)
        if not density_path.exists():
            print("Lol")
            continue
        os.makedirs(OUTPUT_FOLDER / density, exist_ok=True)

        images = list(density_path.glob("*.png"))
        print(len(images))
        

        for img_path in images:
            print(f"Current Image: {img_path}")
            raw_img = cv2.imread(str(img_path))
            #normalized_img = normalize_image_rgb(raw_img, method='min_max')
            gray_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2GRAY)
            if density == '80x':
                normalized_image = normalize_image_rgb(gray_img,method='clahe')
            else:
                normalized_image = normalize_image_rgb(gray_img, method='min_max')
            print(np.median(normalized_image))
            #normalized_image[normalized_image > 150] = 255
            plt.imshow(normalized_image)
            plt.show()
            plt.close()
            cv2.imwrite(str(OUTPUT_FOLDER / density / Path(img_path).name), normalized_image)
