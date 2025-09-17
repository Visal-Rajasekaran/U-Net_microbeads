from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import torchvision.transforms as T
from model_v8 import *

# Device config
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset2", "images_selected_normalized")
RESULTS_FOLDER = os.path.join(BASE_DIR, "synthetic_data_generation", "sdg_3_model_v8_test_on_original_results")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Define transform
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

# Load model
model_path = os.path.join(BASE_DIR, "weights_outputs", "weights_sdg_3_model_v8_heat_and_seg.pth")
model = NANO_PICS().to(device)
model.load_state_dict(torch.load(model_path, map_location=device,weights_only=True))
model.eval()

def visualize_single_image(image_path, index=0):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_pred,heat_pred = model(img_tensor)

    # Post-process
    seg_np = seg_pred.squeeze().cpu().numpy()
    heat_np = heat_pred.squeeze().cpu().numpy()
    #thresholded_seg_np = (seg_np >= 0.9).astype(np.uint8)
    img_np = np.array(img.resize((512, 512)))

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(img_np)
    axs[0].set_title("Input Image")

    axs[1].imshow(seg_np, cmap='gray')
    axs[1].set_title("Raw Predicted Mask")

    axs[2].imshow(heat_np.astype(np.uint8), cmap='gray')
    axs[2].set_title("Heatmap Predicted Mask")

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    base_filename = os.path.basename(image_path).split('.')[0]
    plt.savefig(os.path.join(RESULTS_FOLDER, f"{base_filename}_prediction.png"))
    plt.close()

# Run inference on all images in IMAGE_FOLDER
from pathlib import Path

image_dir = Path(IMAGE_FOLDER)
image_paths = sorted(image_dir.rglob("*"))  # Recursively gets all files

for idx, img_path in enumerate(image_paths):
    if img_path.is_file():
        print(f"Inferencing on {img_path}...")
        visualize_single_image(str(img_path), index=idx)
