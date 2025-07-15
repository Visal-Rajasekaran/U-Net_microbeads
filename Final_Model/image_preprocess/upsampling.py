
import os
import numpy as np
import cv2

def upsample(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)# if image.ndim == 3 else image
    resized_img = cv2.resize(gray, (0, 0), fx=20.0, fy=20.0, interpolation=cv2.INTER_CUBIC)
    print(resized_img.shape)
    return resized_img


def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    img_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ]

    for fname in img_files:
        in_path = os.path.join(input_folder, fname)
        out_path = os.path.join(output_folder, fname)

        image = cv2.imread(in_path)
        
        if image is None:
            print(f"[WARN] Could not load: {in_path}")
            continue
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        #gray= cv2.bitwise_not(gray)

        filtered = upsample(image)
        cv2.imwrite(out_path, filtered)
        print(f"[OK] Filtered: {fname}")
from pathlib import Path
# === RUN ===
if __name__ == "__main__":
    BASE_DIR = BASE_DIR = Path(__file__).resolve().parents[1]  # 1 levels up
    INPUT_FOLDER = os.path.join(BASE_DIR, "dataset", "selected_80x")
    OUTPUT_FOLDER = os.path.join(BASE_DIR,"dataset", "upsample_grayscale_selected_80x")

    process_folder(INPUT_FOLDER, OUTPUT_FOLDER)
