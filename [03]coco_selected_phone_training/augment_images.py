import cv2
import numpy as np
import os
import glob
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images")

# Output folders
OUTPUT_DIRS = {
    "all_permutations": os.path.join(BASE_DIR, "dataset", "augmented_images_all_permutations"),
    "balanced_random": os.path.join(BASE_DIR, "dataset", "augmented_images_balanced_random"),
    "directional_only": os.path.join(BASE_DIR, "dataset", "augmented_images_directional_only")
}
for path in OUTPUT_DIRS.values():
    os.makedirs(path, exist_ok=True)

def blur_image(image):
    downscale_factor = 0.25
    small = cv2.resize(image, (0, 0), fx=downscale_factor, fy=downscale_factor, interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

def apply_vignette(image):
    h, w = image.shape[:2]
    Y, X = np.ogrid[:h, :w]
    center_x, center_y = w / 2, h / 2
    radius = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
    vignette_strength = random.uniform(0.3, 0.6)
    vignette = np.clip(1 - (radius / radius.max())**2, vignette_strength, 1)
    return (image.astype(np.float32) * vignette[..., np.newaxis]).astype(np.uint8)

def apply_directional_lighting(image, direction):
    h, w = image.shape[:2]
    if direction == "left":
        gradient = np.tile(np.linspace(1.3, 0.5, w), (h, 1))
    else:
        gradient = np.tile(np.linspace(0.5, 1.3, w), (h, 1))
    return (image * gradient[..., np.newaxis])

# Load images
image_paths = glob.glob("C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[[]03[]]coco_selected_phone_training\\dataset\\images\\*.png")
print(f"[INFO] Found {len(image_paths)} images.")

for path in image_paths:
    img = cv2.imread(path)
    if img is None:
        print(f"[WARN] Skipping unreadable image: {path}")
        continue
    filename = os.path.basename(path)
    
    # Common preprocessing (blurring)
    blurred = blur_image(img)

    # === all_permutations ===
    cv2.imwrite(os.path.join(OUTPUT_DIRS["all_permutations"], f"{filename[:-4]}_vignette.png"),
                apply_vignette(blurred))
    cv2.imwrite(os.path.join(OUTPUT_DIRS["all_permutations"], f"{filename[:-4]}_leftlight.png"),
                apply_directional_lighting(blurred, "left"))
    cv2.imwrite(os.path.join(OUTPUT_DIRS["all_permutations"], f"{filename[:-4]}_rightlight.png"),
                apply_directional_lighting(blurred, "right"))

    # === balanced_random ===
    choice = random.choice(["vignette", "left", "right"])
    if choice == "vignette":
        output = apply_vignette(blurred)
    else:
        output = apply_directional_lighting(blurred, direction=choice)
    cv2.imwrite(os.path.join(OUTPUT_DIRS["balanced_random"], filename), output)

    # === directional_only ===
    direction = random.choice(["left", "right"])
    dir_output = apply_directional_lighting(blurred, direction)
    cv2.imwrite(os.path.join(OUTPUT_DIRS["directional_only"], filename), dir_output)

print("[DONE] Augmentations completed.")
