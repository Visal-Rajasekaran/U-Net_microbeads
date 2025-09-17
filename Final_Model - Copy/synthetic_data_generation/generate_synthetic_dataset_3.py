import numpy as np
import cv2
import os
from pathlib import Path
from skimage import exposure
import shutil
import random
import csv
from datetime import datetime

# -------------------- Configuration -------------------- #
IMG_SIZE = 512
CROP_SIZE = 7
STRIDE = 1
SAVE_BEAD_LIBRARIES = True  # Set to False to disable saving
apply_blur = False

BASE_FOLDER = Path(__file__).parent
INPUT_ROOT = BASE_FOLDER.parent / "dataset2" / "images_selected"
BEAD_FOLDER = BASE_FOLDER / "bead_library"
BACKGROUND_FOLDER = BASE_FOLDER / "background_library"
SAMPLE_FOLDER = BASE_FOLDER / "synthetic_images_4_grayscale"
MASK_FOLDER = BASE_FOLDER / "masks_4"
CSV_LOG = SAMPLE_FOLDER / "synthetic_centroids_4.csv"

#shutil.rmtree(SAMPLE_FOLDER, ignore_errors=True)
#shutil.rmtree(MASK_FOLDER,ignore_errors=True)
#os.remove(CSV_LOG)
SAMPLE_FOLDER.mkdir(exist_ok=True)
MASK_FOLDER.mkdir(exist_ok=True)

if not CSV_LOG.exists():
    with open(CSV_LOG, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'image_filename', 'bead_id', 'category_id', 'x', 'y', 'area', 'approx_radius'])

MAX_BEADS = 1000
MAX_BACKGROUNDS = 1000

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
        # Convert to LAB and apply CLAHE on L channel
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    elif method == 'histogram':
        norm = image.astype(np.float32)
        imin, imax = np.percentile(norm, (2, 98))
        if imax - imin < 1e-5: return image
        norm = (norm - imin) / (imax - imin) * 255
        return np.clip(norm, 0, 255).astype(np.uint8)
    elif method == 'min_max':    
        norm_image = np.zeros_like(image, dtype=np.float32)
        #image = cv2.RGB2
        for c in range(3):  # For each color channel
            channel = image[:, :, c].astype(np.float32)
            min_val = channel.min()
            max_val = channel.max()
            if max_val > min_val:  # Avoid division by zero
                norm_image[:, :, c] = (channel - min_val) / (max_val - min_val)
            else:
                norm_image[:, :, c] = 0  # flat color
        return (norm_image * 255).astype(np.uint8)

    else:
        return image  # No normalization


def has_bead_signature(crop, edge_thresh_bead, bright_thresh_bead, edge_thresh_bg,bright_thresh_bg, bead=True):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    center = gray.shape[0] // 2
    center_region = gray[center-2:center+2, center-2:center+2]
    edge_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if bead:
        return (edge_pixels.mean() - center_region.mean()) > edge_thresh_bead and gray.mean() < bright_thresh_bead
    else:
        return gray.mean() > bright_thresh_bg and abs(edge_pixels.mean() - center_region.mean()) < edge_thresh_bg

# -------------------- Library Creation -------------------- #
def create_bead_library(image, ref_brightness, edge_thresh_bead, bright_thresh_bead,edge_thresh_bg,bright_thresh_bg):
    shutil.rmtree(BEAD_FOLDER, ignore_errors=True)
    shutil.rmtree(BACKGROUND_FOLDER, ignore_errors=True)
    BEAD_FOLDER.mkdir(exist_ok=True)
    BACKGROUND_FOLDER.mkdir(exist_ok=True)
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #image = normalize_image_rgb(image, method="min_max")
    image_brightness = np.median(image)
    brightness_offset = image_brightness - ref_brightness

    edge_thresh_bead = 30 + brightness_offset*0
    bright_thresh_bead = 170 + brightness_offset*0
    edge_thresh_bg = 10 + brightness_offset*0
    bright_thresh_bg = 200 + brightness_offset*0
    '''brightness_offset = np.clip(image_brightness - ref_brightness, -30, 30)

    edge_thresh_bead = (0.4 * image.std()) + brightness_offset
    bright_thresh_bead = 80 + brightness_offset
    edge_thresh_bg = (0.1 * image.std()) + brightness_offset
    bright_thresh_bg = 125 + brightness_offset'''


    h, w, _ = image.shape
    bead_count = 0
    bg_count = 0
    for y in range(0, h - CROP_SIZE + 1, STRIDE):
        for x in range(0, w - CROP_SIZE + 1, STRIDE):
            patch = image[y:y+CROP_SIZE, x:x+CROP_SIZE, :]
            if has_bead_signature(patch, edge_thresh_bead, bright_thresh_bead,edge_thresh_bg,bright_thresh_bg, bead=True) and bead_count < MAX_BEADS:
                cv2.imwrite(str(BEAD_FOLDER / f"bead_{bead_count}.png"), patch)
                bead_count += 1
            elif has_bead_signature(patch, edge_thresh_bead, bright_thresh_bead,edge_thresh_bg,bright_thresh_bg, bead=False) and bg_count < MAX_BACKGROUNDS:
                cv2.imwrite(str(BACKGROUND_FOLDER / f"bg_{bg_count}.png"), patch)
                bg_count += 1
            if bead_count >= MAX_BEADS and bg_count >= MAX_BACKGROUNDS:
                return

# -------------------- Centroid Generation -------------------- #
def generate_irregular_clusters(image_size, num_clusters, cluster_size_range, min_distance):
    centroids = []
    occupied = set()
    def is_valid(x, y):
        return 0 <= x < image_size and 0 <= y < image_size and (x, y) not in occupied

    for _ in range(num_clusters):
        cluster_size = random.randint(*cluster_size_range)
        seed_x = random.randint(0, image_size - 1)
        seed_y = random.randint(0, image_size - 1)
        cluster = [(seed_x, seed_y)]
        occupied.add((seed_x, seed_y))
        centroids.append((seed_x, seed_y))
        attempts = 0
        while len(cluster) < cluster_size and attempts < cluster_size * 10:
            base_x, base_y = random.choice(cluster)
            dx = random.randint(-min_distance, min_distance)
            dy = random.randint(-min_distance, min_distance)
            new_x = base_x + dx
            new_y = base_y + dy
            if is_valid(new_x, new_y):
                cluster.append((new_x, new_y))
                occupied.add((new_x, new_y))
                centroids.append((new_x, new_y))
            attempts += 1

    return centroids

# -------------------- Alpha Mask -------------------- #
def create_bead_alpha_mask(bead_crop):
    h, w = bead_crop.shape[:2]
    y, x = np.ogrid[:h, :w]
    cx, cy = w // 2, h // 2
    r = min(h, w) // 2
    mask = (x - cx)**2 + (y - cy)**2 <= r**2
    alpha = np.zeros((h, w), dtype=np.float32)
    alpha[mask] = 1.0
    return alpha[..., None]

# -------------------- Blending -------------------- #
def blend_bead_at_position(base_img, bead_crop, position):
    x, y = position
    h, w, _ = bead_crop.shape
    dy = min(h, base_img.shape[0] - y)
    dx = min(w, base_img.shape[1] - x)
    if dy <= 0 or dx <= 0:
        return base_img
    bead_crop = bead_crop[:dy, :dx]
    roi = base_img[y:y+dy, x:x+dx]
    alpha = create_bead_alpha_mask(bead_crop[:dy, :dx])
    blended = alpha * bead_crop.astype(np.float32) + (1 - alpha) * roi.astype(np.float32)
    base_img[y:y+dy, x:x+dx] = blended.astype(np.uint8)
    return base_img

# -------------------- Augmentation -------------------- #
def augment_image(img):
    return cv2.GaussianBlur(img, (7, 7), 2)

# -------------------- Synthetic Generation -------------------- #
def load_library(folder):
    return [cv2.imread(str(p)) for p in folder.glob("*.png")]

def generate_synthetic_image(image_id, num_clusters, cluster_size_range):
    base_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    bead_lib = load_library(BEAD_FOLDER)
    bg_lib = load_library(BACKGROUND_FOLDER)
    
    if not bg_lib:
        print(f"[WARNING] Empty background library for image_id {image_id}, skipping.")
        return

    for y in range(0, IMG_SIZE, CROP_SIZE):
        for x in range(0, IMG_SIZE, CROP_SIZE):
            patch = random.choice(bg_lib)
            dy = min(CROP_SIZE, IMG_SIZE - y)
            dx = min(CROP_SIZE, IMG_SIZE - x)
            patch = patch[:dy, :dx, :]
            base_img[y:y+dy, x:x+dx] = patch

    centroids = generate_irregular_clusters(
        image_size=IMG_SIZE,
        num_clusters=num_clusters,
        cluster_size_range=cluster_size_range,
        min_distance=7
    )

    bead_radius = CROP_SIZE // 2
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    image_filename = f"synthetic_80x_{image_id}_{timestamp}.png"

    with open(CSV_LOG, mode='a', newline='') as f:
        writer = csv.writer(f)
        for bead_id, pos in enumerate(centroids, 1):
            x, y = pos
            bead = random.choice(bead_lib)
            base_img = blend_bead_at_position(base_img, bead, (x, y))
            cv2.circle(mask, (x, y), bead_radius, 255, thickness=-1)
            area = np.pi * (bead_radius**2)
            writer.writerow([image_id + 1, image_filename, bead_id, 1, x, y, int(area), bead_radius])

    if apply_blur == True:
        base_img = augment_image(base_img) 
    cv2.imwrite(str(SAMPLE_FOLDER / image_filename), base_img)
    cv2.imwrite(str(MASK_FOLDER / image_filename), mask)

# -------------------- Main -------------------- #
if __name__ == "__main__":
    ref_img = cv2.imread("C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\Final_Model\\dataset2\\images_selected\80x\\80x_2025-05-15_02-05-00_001_grid_r0_c2.png")
    ref_brightness = ref_img.mean()

    density_config = {
        "80x": (1000, (1, 15), 7),
        "320x": (500, (1, 4), 8),
        "640x": (300, (1, 2), 9),
        "1280x": (100, (1, 2), 10),
        "2560x": (50, (1, 2), 10),
        "5120x": (20, (1, 2), 10),
        "10240x": (10, (1, 1), 10),
    }
    edge_thresh_bead, bright_thresh_bead,edge_thresh_bg,bright_thresh_bg = 40,120,10,160

    img_count = 0
    for density, (num_clusters, cluster_range, bead_size) in density_config.items():
        density_path = INPUT_ROOT / density
        if not density_path.exists():
            continue
        CROP_SIZE = bead_size
        if density == "80x":
            apply_blur = True
            edge_thresh_bead, bright_thresh_bead,edge_thresh_bg,bright_thresh_bg = 60,100,10,200
        else:
            apply_blur = False
            edge_thresh_bead, bright_thresh_bead,edge_thresh_bg,bright_thresh_bg = 60,100,10,200

        images = list(density_path.glob("*.png"))
        selected_images = random.sample(images, min(5, len(images)))

        for img_path in selected_images:
            print(f"Current Image: {img_path}")
            raw_img = cv2.imread(str(img_path))
            create_bead_library(raw_img, ref_brightness,edge_thresh_bead, bright_thresh_bead,edge_thresh_bg,bright_thresh_bg)
            for _ in range(2):
                generate_synthetic_image(img_count, num_clusters, cluster_range)
                img_count += 1
