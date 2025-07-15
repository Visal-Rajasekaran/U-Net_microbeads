import cv2
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os
# Load image
def pca(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img_rgb.shape

    # Reshape to (num_pixels, 3)
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)

    # Center the data
    pixels_mean = pixels.mean(axis=0)
    pixels_centered = pixels - pixels_mean

    # PCA
    pca = PCA(n_components=3)
    pca.fit(pixels_centered)
    transformed = pca.transform(pixels_centered)

    # Analyze how much each PC aligns with the noisy red channel
    print("PCA components (each row = one PC direction):")
    print(pca.components_)

    # Option 1: Drop PC most aligned with Red channel
    red_axis = np.array([1, 0, 0])
    dot_products = np.abs(pca.components_ @ red_axis)
    drop_pc_idx = np.argmax(dot_products)

    print(f"Dropping PC {drop_pc_idx} (most aligned with Red)")

    # Zero out that component
    transformed[:, drop_pc_idx] = 0

    # Reconstruct image
    reconstructed = pca.inverse_transform(transformed)
    reconstructed += pixels_mean  # re-add mean
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    img_denoised = reconstructed.reshape(h, w, 3)

    # Show result
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img_rgb)
    axs[0].set_title("Original")
    axs[1].imshow(img_denoised)
    axs[1].set_title("Red-suppressed via PCA")
    for ax in axs: ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.close()

def process_folder(input_folder, output_folder, cutoff_percent=0.3):
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

        image[..., 2] = 0

        cv2.imwrite(out_path, image)
        print(f"[OK] Filtered: {fname}")

from pathlib import Path
# === RUN ===
if __name__ == "__main__":
    BASE_DIR = BASE_DIR = Path(__file__).resolve().parents[1]  # 1 levels up
    INPUT_FOLDER = os.path.join(BASE_DIR, "dataset", "selected_80x")
    OUTPUT_FOLDER = os.path.join(BASE_DIR,"dataset", "channel_3_removed_selected_80x")

    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, cutoff_percent=1.0)