import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gc
import os



# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COCO_JSON_PATH = os.path.join(BASE_DIR, "annotations.json")
IMAGE_FOLDER = os.path.join(BASE_DIR, "dataset", "images")
RESULTS_FOLDER = os.path.join(BASE_DIR, "results")
CENTROIDS_FOLDER = os.path.join(BASE_DIR, "dataset", "centroids")
MASKS_FOLDER = os.path.join(BASE_DIR, "dataset", "masks")

'''class HeatmapRegressionNet(nn.Module):
    def __init__(self):
        super(HeatmapRegressionNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return torch.sigmoid(x)  # output shape: (B, 1, H, W)'''


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.dec4 = ConvBlock(1024, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.dec3 = ConvBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.dec2 = ConvBlock(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.dec1 = ConvBlock(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        u4 = self.up4(b)
        a4 = self.att4(g=u4, x=s4)
        d4 = self.dec4(torch.cat([u4, a4], dim=1))

        u3 = self.up3(d4)
        a3 = self.att3(g=u3, x=s3)
        d3 = self.dec3(torch.cat([u3, a3], dim=1))

        u2 = self.up2(d3)
        a2 = self.att2(g=u2, x=s2)
        d2 = self.dec2(torch.cat([u2, a2], dim=1))

        u1 = self.up1(d2)
        a1 = self.att1(g=u1, x=s1)
        d1 = self.dec1(torch.cat([u1, a1], dim=1))

        return torch.sigmoid(self.out(d1))

    def forward(self, x):
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        b = self.bottleneck(self.pool(s4))

        d4 = self.dec4(torch.cat([self.up4(b), s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return torch.sigmoid(self.out(d1))
    
class NANO_PICS(nn.Module):
    def __init__(self):
        super().__init__()
        self.segmentation_net = UNet(in_channels=3, out_channels=1)
        self.centroid_net = UNet(in_channels=3, out_channels=1)

    def forward(self, x):
        seg_out = self.segmentation_net(x)
        centroid_out = self.centroid_net(x)
        return seg_out, centroid_out

def generate_heatmap(image_shape, points, radii, default_sigma=4):
    heatmap = np.zeros(image_shape, dtype=np.float32)
    for (x, y), radius in zip(points, radii):
        sigma = radius / 2 if radius else default_sigma #radius/2 seems better than typical /3
        tmp = np.zeros(image_shape, dtype=np.float32)
        tmp[int(y), int(x)] = 1
        tmp = cv2.GaussianBlur(tmp, (0,0), sigma)
        tmp /= tmp.max()
        heatmap = np.maximum(heatmap, tmp)
    return heatmap

class CentroidDataset(Dataset):
    def __init__(self, image_root, image_size=(512, 512), transform=None):
        self.image_root = image_root
        self.image_size = image_size
        self.transform = transform
        self.image_files = [
            fname for fname in os.listdir(image_root)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))
        ]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_root, filename)

        image = Image.open(image_path).convert("RGB").resize(self.image_size)

        if self.transform:
            image = self.transform(image)
        else:
            image = T.ToTensor()(image)  # shape: [3, H, W]

        return {
            'image': image,
            'filename': filename
        }


from sklearn.mixture import GaussianMixture

def fit_gaussians_to_heatmap(heatmap, n_components=20, threshold=0.2):
    """
    Fit a GMM to the nonzero pixels of the heatmap.
    
    Args:
        heatmap (np.array): 2D array (H, W) from model output.
        n_components (int): Number of Gaussians to fit (e.g. number of expected centroids).
        threshold (float): Pixel intensity threshold for masking noise.

    Returns:
        means (np.array): Array of shape [n_components, 2] with (y, x) coordinates.
        covariances (np.array): Covariance matrices [n_components, 2, 2]
    """
    # Threshold and extract weighted coordinates
    coords = np.column_stack(np.nonzero(heatmap > threshold))  # shape (N, 2)
    weights = heatmap[coords[:, 0], coords[:, 1]]  # pixel intensities
    print(len(coords))
    if len(coords) < n_components:
        print("Warning: not enough points to fit GMM.")
        return None, None

    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(coords)

    return gmm.means_, gmm.covariances_

from scipy.ndimage import maximum_filter, label

def count_local_maxima(heatmap, threshold=0.5, neighborhood_size=3):
    """
    Counts local peaks in the heatmap above a certain threshold.
    """
    # Apply threshold
    heatmap = (heatmap > threshold) * heatmap

    # Find local maxima
    neighborhood = maximum_filter(heatmap, size=neighborhood_size) == heatmap
    labeled, num_objects = label(neighborhood)

    return num_objects

def visualize_outputs(model_path, dataset, index=0, device='cuda'):

    # Load model
    model = NANO_PICS().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    # Get sample
    sample = dataset[index]
    image_tensor = sample['image'].unsqueeze(0).to(device)  # [1, 3, H, W]
    filename = sample['filename']

    with torch.no_grad():
        seg_pred, centroid_pred = model(image_tensor)

    # Convert predictions to numpy
    image_np = image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    seg_np = seg_pred.squeeze().cpu().numpy()
    centroid_np = centroid_pred.squeeze().cpu().numpy()

    # Threshold for segmentation mask
    seg_binary = (seg_np >= 0.5).astype(np.uint8)

    # Count blobs from heatmap
    num_peaks = count_local_maxima(centroid_np, threshold=0.1)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(image_np)
    axs[0].set_title("Input Image")
    axs[1].imshow(seg_binary, cmap="gray")
    axs[1].set_title("Predicted Mask (Thresholded)")
    axs[2].imshow(centroid_np, cmap="hot")
    axs[2].set_title(f"Predicted Heatmap (Peaks: {num_peaks})")

    for ax in axs:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig("Test_2_augmented_v5_1.png")
    plt.close()

    print(f"Saved visualization for {filename} with {num_peaks} detected peaks.")





def estimate_best_gmm_components(heatmap, max_components=300, threshold=0.01):
    coords = np.column_stack(np.nonzero(heatmap > threshold))
    lowest_bic = np.inf
    best_gmm = None

    for n in range(499, max_components + 1):
        gmm = GaussianMixture(n_components=n, covariance_type='full')
        gmm.fit(coords)
        bic = gmm.bic(coords)
        if bic < lowest_bic:
            lowest_bic = bic
            best_gmm = gmm
        print("components: ", n)

    return best_gmm.means_, best_gmm.covariances_

model_path = os.path.join(BASE_DIR, "weights_phone_augmented_v5.pth")
train_set = CentroidDataset(IMAGE_FOLDER)
visualize_outputs(model_path, train_set, index=1)


