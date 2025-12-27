import cv2
import numpy as np
import pandas as pd
from pathlib import Path

BASE_FOLDER = Path(__file__).parent
INPUT_ROOT = BASE_FOLDER.parent / "dataset2" / "images_selected"
BEAD_FOLDER = BASE_FOLDER / "bead_library"
BACKGROUND_FOLDER = BASE_FOLDER / "background_library"
SAMPLE_FOLDER = BASE_FOLDER / "synthetic_images_4"
MASK_FOLDER = BASE_FOLDER / "masks_4_circular"
CSV_LOG = SAMPLE_FOLDER / "synthetic_centroids_4.csv"

def regenerate_masks(csv_file, output_folder, image_size=512):
    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)

    df = pd.read_csv(csv_file)

    for image_filename, group in df.groupby("image_filename"):
        mask = np.zeros((image_size, image_size), dtype=np.uint8)

        for _, row in group.iterrows():
            x, y, r = int(row["x"]), int(row["y"]), int(row["approx_radius"])
            cv2.circle(mask, (x, y), r, 255, thickness=-1, lineType=cv2.LINE_AA)

        cv2.imwrite(str(output_folder / image_filename), mask)

if __name__ == "__main__":
    csv_file = CSV_LOG  # path to your CSV
    output_folder = MASK_FOLDER                  # new folder for regenerated masks
    regenerate_masks(csv_file, output_folder)
