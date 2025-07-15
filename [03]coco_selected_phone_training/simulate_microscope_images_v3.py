import os
import cv2
import numpy as np
import glob
import random

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "dataset", "augmented_images_v3")
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === TRANSFORM FUNCTIONS ===
def simulate_blur(image, factor=0.25):
    h, w = image.shape[:2]
    small = cv2.resize(image, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)

def darken_image(image, factor=0.5):
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)

def reduce_contrast(image, factor=0.5):
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)

def add_color_noise(image, sigma=15):
    noise = np.random.normal(0, sigma, image.shape).astype(np.float32)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def apply_blue_green_tint(image, red_scale=0.85, green_scale=1.05, blue_scale=1.10):
    tinted = image.astype(np.float32)
    tinted[..., 0] *= blue_scale
    tinted[..., 1] *= green_scale
    tinted[..., 2] *= red_scale
    return np.clip(tinted, 0, 255).astype(np.uint8)

def increase_contrast(image, alpha=1.3):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def contrast_stretch(image, low_perc=5, high_perc=95):
    if image.ndim == 3:
        channels = cv2.split(image)
        out_channels = []
        for ch in channels:
            lo = np.percentile(ch, low_perc)
            hi = np.percentile(ch, high_perc)
            stretched = np.clip((ch - lo) * 255.0 / (hi - lo + 1e-5), 0, 255)
            out_channels.append(stretched.astype(np.uint8))
        return cv2.merge(out_channels)
    else:
        lo = np.percentile(image, low_perc)
        hi = np.percentile(image, high_perc)
        return np.clip((image - lo) * 255.0 / (hi - lo + 1e-5), 0, 255).astype(np.uint8)

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

def simulate_magnification(image, factor=1.2):
    """
    Simulates magnification by scaling around the image center and resizing back.
    - factor > 1: zoom in
    - factor < 1: zoom out
    Output has the same shape as input.
    """
    h, w = image.shape[:2]
    new_h, new_w = int(h / factor), int(w / factor)

    # Ensure dimensions are valid
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    # Center crop
    start_y = (h - new_h) // 2
    start_x = (w - new_w) // 2
    cropped = image[start_y:start_y + new_h, start_x:start_x + new_w]

    # Resize back to original size
    magnified = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
    return magnified



def degrade_to_microscope_style(image):
    #image = darken_image(image, factor=0.5)
    #image = increase_contrast(image, alpha=4.0)                   # Enhance local bead contrast
    #image = contrast_stretch(image, low_perc=15,high_perc=85)
    #image = simulate_magnification(image, factor = 1.4)
    image = apply_clahe(image)
    image = simulate_blur(image, factor=1)
    image = add_color_noise(image, sigma=5)
    image = apply_blue_green_tint(image, red_scale=0.85, green_scale=0.95, blue_scale=1.30)
    return image

def apply_directional_darkening(image, direction="left", strength=0.8):
    h, w = image.shape[:2]
    if direction == "left":
        gradient = np.tile(np.linspace(1 - strength, 1.0, w), (h, 1))
    elif direction == "right":
        gradient = np.tile(np.linspace(1.0, 1 - strength, w), (h, 1))
    else:
        raise ValueError("Direction must be 'left' or 'right'")
    if image.ndim == 3:
        gradient = gradient[..., np.newaxis]
    return np.clip(image.astype(np.float32) * gradient, 0, 255).astype(np.uint8)

def apply_radial_darkening(image, strength=0.5):
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w / 2, h / 2
    radius = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    mask = 1 - strength * (radius / radius.max())**2
    mask = np.clip(mask, 0, 1)
    if image.ndim == 3:
        mask = mask[..., np.newaxis]
    return np.clip(image.astype(np.float32) * mask, 0, 255).astype(np.uint8)

# === MAIN LOOP ===
image_paths = glob.glob("C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[[]03[]]coco_selected_phone_training\\dataset\\images\\*.png")
print(f"[INFO] Found {len(image_paths)} images to process.")
random.shuffle(image_paths)

num_images = len(image_paths)

for idx, path in enumerate(image_paths):
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Skipping unreadable image: {path}")
        continue

    degraded = degrade_to_microscope_style(img)

    # Apply directional darkening based on proportion
    if idx < num_images / 3:
        degraded = apply_directional_darkening(degraded, direction="left")
    elif idx < 2 * num_images / 3:
        degraded = apply_directional_darkening(degraded, direction="right")
    # else: no darkening

    filename = os.path.basename(path)
    output_path = os.path.join(OUTPUT_FOLDER, filename)
    cv2.imwrite(output_path, degraded)
    print(f"[OK] Saved degraded: {filename}")

print(f"[DONE] All images saved to: {OUTPUT_FOLDER}")
