import os
import cv2
from pathlib import Path
from tqdm import tqdm

# Config
DILUTIONS = ["80x", "320x", "640x", "1280x", "2560x", "5120x", "10240x"]
INPUT_ROOT = Path("dataset")
OUT_IMAGE_ROOT = Path("dataset2") / "synthetic_data" / "images"
OUT_MASK_ROOT = Path("dataset2") / "synthetic_data" / "masks"

# Import from your synthetic generator script
from generate_synthetic_image import create_bead_library, generate_synthetic_image

def generate_for_dilution(dilution_folder):
    image_paths = sorted(list(dilution_folder.glob("*.png")))[:4]  # pick 4 images
    for img_path in image_paths:
        img = cv2.imread(str(img_path))
        create_bead_library(img)

        for i in range(5):  # 5 synthetic per image
            synth_img, mask, _ = generate_synthetic_image()

            out_img_name = img_path.stem + f"_synth{i}.png"
            out_mask_name = img_path.stem + f"_synth{i}_mask.png"

            cv2.imwrite(str(OUT_IMAGE_ROOT / out_img_name), synth_img)
            cv2.imwrite(str(OUT_MASK_ROOT / out_mask_name), mask)

def main():
    for dil in DILUTIONS:
        dil_folder = INPUT_ROOT / "images_selected" / dil
        (OUT_IMAGE_ROOT / dil).mkdir(parents=True, exist_ok=True)
        (OUT_MASK_ROOT / dil).mkdir(parents=True, exist_ok=True)
        generate_for_dilution(dil_folder)

if __name__ == "__main__":
    main()
