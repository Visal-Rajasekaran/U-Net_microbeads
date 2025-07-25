import numpy as np
import cv2
import os
from pathlib import Path
from skimage import exposure
import shutil
import random
import csv

# -------------------- Configuration -------------------- #
IMG_SIZE = 512
CROP_SIZE = 7
STRIDE = 1
NUM_SYNTHETIC_IMAGES = 2

BASE_FOLDER = Path(__file__).parent
IMAGE_FOLDER = BASE_FOLDER.parent / "dataset2" / "images_selected" / "80x"
BEAD_FOLDER = BASE_FOLDER / "bead_library"
BACKGROUND_FOLDER = BASE_FOLDER / "background_library"
SAMPLE_FOLDER = BASE_FOLDER / "synthetic_images"
MASK_FOLDER = BASE_FOLDER / "masks"
CSV_LOG = BASE_FOLDER / "synthetic_centroids.csv"

for folder in [BEAD_FOLDER, BACKGROUND_FOLDER]:
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(exist_ok=True)
SAMPLE_FOLDER.mkdir(exist_ok=True)
MASK_FOLDER.mkdir(exist_ok=True)

if not CSV_LOG.exists():
    with open(CSV_LOG, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'image_filename', 'bead_id', 'category_id', 'x', 'y', 'area', 'approx_radius'])

MAX_BEADS = 1000
MAX_BACKGROUNDS = 1000

# -------------------- Bead Detection -------------------- #
def has_bead_signature(crop, threshold=0.1, bead=True):
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    center = gray.shape[0] // 2
    center_region = gray[center-2:center+2, center-2:center+2]
    edge_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
    if bead:
        return (edge_pixels.mean() - center_region.mean()) > threshold or gray.mean() < 80
    else:
        return gray.mean() > 125 and abs(edge_pixels.mean() - center_region.mean()) < threshold

# -------------------- Library Creation -------------------- #
def create_bead_library(image):
    shutil.rmtree(BEAD_FOLDER, ignore_errors=True)
    shutil.rmtree(BACKGROUND_FOLDER, ignore_errors=True)
    BEAD_FOLDER.mkdir(exist_ok=True)
    BACKGROUND_FOLDER.mkdir(exist_ok=True)

    h, w, _ = image.shape
    bead_count = 0
    bg_count = 0
    for y in range(0, h - CROP_SIZE + 1, STRIDE):
        for x in range(0, w - CROP_SIZE + 1, STRIDE):
            patch = image[y:y+CROP_SIZE, x:x+CROP_SIZE, :]
            if has_bead_signature(patch, threshold=40, bead=True) and bead_count < MAX_BEADS:
                cv2.imwrite(str(BEAD_FOLDER / f"bead_{bead_count}.png"), patch)
                bead_count += 1
            elif has_bead_signature(patch, threshold=10, bead=False) and bg_count < MAX_BACKGROUNDS:
                cv2.imwrite(str(BACKGROUND_FOLDER / f"bg_{bg_count}.png"), patch)
                bg_count += 1
            if bead_count >= MAX_BEADS and bg_count >= MAX_BACKGROUNDS:
                return
    print(f"Bead Count: {bead_count}")

# -------------------- Centroid Generation -------------------- #
def generate_irregular_clusters(image_size=512, num_clusters=5, cluster_size_range=(10, 50), min_distance=3):
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

def generate_synthetic_image(image_id):
    base_img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    bead_lib = load_library(BEAD_FOLDER)
    bg_lib = load_library(BACKGROUND_FOLDER)
    for y in range(0, IMG_SIZE, CROP_SIZE):
        for x in range(0, IMG_SIZE, CROP_SIZE):
            patch = random.choice(bg_lib)
            dy = min(CROP_SIZE, IMG_SIZE - y)
            dx = min(CROP_SIZE, IMG_SIZE - x)
            patch = patch[:dy, :dx, :]
            base_img[y:y+dy, x:x+dx] = patch

    centroids = generate_irregular_clusters(
        image_size=IMG_SIZE,
        num_clusters=1000,
        cluster_size_range=(1, 15),
        min_distance=10
    )

    bead_radius = CROP_SIZE // 2
    image_filename = f"synthetic_80x_{image_id}.png"
    with open(CSV_LOG, mode='a', newline='') as f:
        writer = csv.writer(f)
        for bead_id, pos in enumerate(centroids, 1):
            x, y = pos
            bead = random.choice(bead_lib)
            base_img = blend_bead_at_position(base_img, bead, (x, y))
            cv2.circle(mask, (x, y), bead_radius, 255, thickness=-1)
            area = np.pi * (bead_radius**2)
            writer.writerow([image_id + 1, image_filename, bead_id, 1, x, y, int(area), bead_radius])

    base_img = augment_image(base_img)
    cv2.imwrite(str(SAMPLE_FOLDER / image_filename), base_img)
    cv2.imwrite(str(MASK_FOLDER / image_filename), mask)

# -------------------- Main -------------------- #
if __name__ == "__main__":
    all_images = list(IMAGE_FOLDER.glob("*.png"))
    #selected_images = ["C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\Final_Model\\dataset2\\images_selected\\80x\\80x_2025-05-16_00-58-00_001_001_grid_r0_c6.png"]
    selected_images = random.sample(all_images, NUM_SYNTHETIC_IMAGES)
    img_count = 20
    for i, img_path in enumerate(selected_images):
        print(f"Current Image {img_path}")
        raw_img = cv2.imread(str(img_path))
        create_bead_library(raw_img)
        for j in range(10):
            generate_synthetic_image(img_count)
            img_count += 1
        print(f"Generated 5 synthetic images for {img_path.name}")
