import os
import random
import shutil
from pathlib import Path
import numpy as np
import cv2
from PIL import Image


class SyntheticBeadGenerator:
    def __init__(self,
                 original_images_dir,
                 bead_library_dir,
                 output_image_dir,
                 output_mask_dir,
                 num_synthetic_per_original=5,
                 image_size=(512, 512),
                 seed=42):
        self.original_images_dir = original_images_dir
        self.bead_library_dir = bead_library_dir
        self.output_image_dir = output_image_dir
        self.output_mask_dir = output_mask_dir
        self.num_synthetic_per_original = num_synthetic_per_original
        self.image_size = image_size
        self.rng = random.Random(seed)
        self._prepare_dirs()

    def _prepare_dirs(self):
        for d in [self.output_image_dir, self.output_mask_dir]:
            d.mkdir(exist_ok=True, parents=True)

    def _load_image(self, path):
        return np.array(Image.open(path).convert("L"))

    def _load_bead_library(self, dilution_folder):
        bead_dir = self.bead_library_dir / dilution_folder
        return [self._load_image(p) for p in bead_dir.glob("*.png")]

    def _generate_single(self, base_image, bead_library):
        """
        Places a random number of beads from the library on a background.
        """
        height, width = self.image_size
        synthetic = np.zeros((height, width), dtype=np.uint8)
        mask = np.zeros_like(synthetic)

        num_beads = self.rng.randint(5, 20)

        for _ in range(num_beads):
            bead = self.rng.choice(bead_library)
            bh, bw = bead.shape
            max_y = height - bh
            max_x = width - bw

            if max_x <= 0 or max_y <= 0:
                continue  # Skip oversized beads

            y = self.rng.randint(0, max_y)
            x = self.rng.randint(0, max_x)

            region = synthetic[y:y+bh, x:x+bw]
            bead_mask = bead > 0
            region[bead_mask] = bead[bead_mask]
            mask[y:y+bh, x:x+bw][bead_mask] = 255

        return synthetic, mask

    def run(self):
        for img_path in self.original_images_dir.glob("*.png"):
            image_name = img_path.stem
            dilution_folder = img_path.parent.name

            base_image = self._load_image(img_path)
            bead_library = self._load_bead_library(dilution_folder)

            if not bead_library:
                print(f"[WARN] No beads found for {dilution_folder}, skipping {image_name}")
                continue

            for i in range(self.num_synthetic_per_original):
                synthetic, mask = self._generate_single(base_image, bead_library)

                img_out_path = self.output_image_dir / f"{image_name}_{i}.png"
                mask_out_path = self.output_mask_dir / f"{image_name}_{i}.png"

                Image.fromarray(synthetic).save(img_out_path)
                Image.fromarray(mask).save(mask_out_path)

            # Clear bead library explicitly to avoid mixing across images
            del bead_library


if __name__ == "__main__":
    generator = SyntheticBeadGenerator(
        original_images_dir=Path("dataset2") / "images_selected",
        bead_library_dir=Path("dataset2") / "beads",
        output_image_dir=Path("dataset2") / "synthetic_data",
        output_mask_dir=Path("dataset2") / "masks",
        num_synthetic_per_original=5,
    )
    generator.run()
