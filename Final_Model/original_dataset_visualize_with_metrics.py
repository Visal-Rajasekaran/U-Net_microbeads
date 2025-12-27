from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import torchvision.transforms as T
from model_v7_deepgatewith2convolution import *
import time
from pathlib import Path

# ---------------- Device ---------------- #
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

# ---------------- Paths ---------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset2", "images_selected_normalized_3")
MASK_FOLDER = os.path.join(BASE_DIR, "synthetic_data_generation", "masks_4")
RESULTS_FOLDER = os.path.join(BASE_DIR, "synthetic_data_generation", "sdg_4_model_v7_deepgatewith2convolution_with_metrics")
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# ---------------- Transform ---------------- #
transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor()
])

# ---------------- Load Model ---------------- #
model_path = os.path.join(BASE_DIR, "weights_outputs", "weights_sdg_4_model_v7_deepgatewith2convolution.pth")
model = NANO_PICS().to(device)
model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
model.eval()

# ---------------- Metrics ---------------- #
def compute_metrics(pred_mask, true_mask):
    pred_flat = pred_mask.flatten()
    true_flat = true_mask.flatten()

    tp = np.sum((pred_flat == 1) & (true_flat == 1))
    tn = np.sum((pred_flat == 0) & (true_flat == 0))
    fp = np.sum((pred_flat == 1) & (true_flat == 0))
    fn = np.sum((pred_flat == 0) & (true_flat == 1))

    epsilon = 1e-7
    dice = (2 * tp) / (2 * tp + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    iou = tp / (tp + fp + fn + epsilon)

    return {
        "Dice": dice,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "IoU": iou
    }

# ---------------- Visualization & Inference ---------------- #
def visualize_and_metrics(image_path, mask_path, threshold=0.5):
    img = Image.open(image_path).convert("RGB")
    mask_gt = Image.open(mask_path).convert("L")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        seg_pred = model(img_tensor)

    seg_np = seg_pred.squeeze().cpu().numpy()
    seg_bin = (seg_np >= threshold).astype(np.uint8)

    mask_gt_resized = mask_gt.resize((512, 512))
    mask_np = np.array(mask_gt_resized)
    mask_bin = (mask_np > 127).astype(np.uint8)

    # Compute metrics
    metrics = compute_metrics(seg_bin, mask_bin)

    # Plot predicted mask with metrics
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(np.array(img.resize((512, 512))))
    axs[0].set_title("Input Image")
    axs[1].imshow(seg_bin, cmap='gray')
    axs[1].set_title("Predicted Mask\n" + "\n".join([f"{k}: {v:.3f}" for k,v in metrics.items()]))

    for ax in axs.flat:
        ax.axis('off')

    plt.tight_layout()
    base_filename = os.path.basename(image_path).split('.')[0]
    plt.savefig(os.path.join(RESULTS_FOLDER, f"{base_filename}_prediction.png"))
    plt.close()

    return metrics

# ---------------- Run Inference on Dataset ---------------- #
image_paths = sorted(Path(IMAGE_FOLDER).rglob("*"))
mask_paths = sorted(Path(MASK_FOLDER).rglob("*"))

all_metrics = []

for img_path, msk_path in zip(image_paths, mask_paths):
    print(f"Processing {img_path}...")
    start = time.time()
    metrics = visualize_and_metrics(str(img_path), str(msk_path))
    end = time.time()
    print(f"Time: {end-start:.3f}s | Metrics: {metrics}")
    all_metrics.append(metrics)

# Compute mean metrics across dataset
mean_metrics = {k: np.mean([m[k] for m in all_metrics]) for k in all_metrics[0]}
print("Mean Metrics:", mean_metrics)