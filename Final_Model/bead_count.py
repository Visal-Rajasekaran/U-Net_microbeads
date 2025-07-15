import os
import cv2
import numpy as np
from pathlib import Path

def detect_blobs(image):
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 20000
    params.filterByCircularity = True
    params.minCircularity = 0.01
    params.filterByInertia = True
    params.minInertiaRatio = 0.2

    detector = cv2.SimpleBlobDetector_create(params)
    return detector.detect(image)

def generate_segmentation_masks(image_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    for fname in image_files:
        image_path = os.path.join(image_folder, fname)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        keypoints = detect_blobs(img)

        # Create empty mask
        mask = np.zeros(img.shape, dtype=np.uint8)

        for i, kp in enumerate(keypoints, start=1):
            cx, cy = map(int, kp.pt)
            radius = int(kp.size / 2)

            # Draw filled circle (or ellipse)
            cv2.circle(mask, (cx, cy), radius, color=i, thickness=-1)  # i as label ID

        # Optional: convert to connected components if overlapping blobs matter
        num_labels, labels = cv2.connectedComponents(mask)

        # Save the mask (each region with different value)
        save_path = os.path.join(output_folder, f"{os.path.splitext(fname)[0]}_mask.png")
        cv2.imwrite(save_path, labels.astype(np.uint8))
        print(f"[✓] Saved mask for {fname} with {num_labels-1} objects")

# === RUN ===
if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[0]
    IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "channel_2_selected_80x")
    OUTPUT_MASKS = os.path.join(BASE_DIR, "label_generation", "segmentation_masks")
    generate_segmentation_masks(IMAGE_FOLDER, OUTPUT_MASKS)
