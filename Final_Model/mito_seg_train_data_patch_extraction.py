import os
import random
import numpy as np
from tifffile import imread, imwrite

# ---------------- Parameters ---------------- #
image_tif_path = "subvolumes/train_images.tif"     # 3D tif (Z,Y,X) or (Z,Y,X,3)
mask_tif_path = "subvolumes/train_masks.tif"       # matching 3D mask tif

save_img_dir = "data/images"
save_mask_dir = "data/masks"

crop_size = 256
num_crops = 100       # total 2D crops
random_seed = 42
os.makedirs(save_img_dir, exist_ok=True)
os.makedirs(save_mask_dir, exist_ok=True)
random.seed(random_seed)

# ---------------- Load 3D stacks ---------------- #
print("Loading image and mask stacks...")
img_stack = imread(image_tif_path)   # shape: (Z, Y, X, [C]) or (Z, Y, X)
mask_stack = imread(mask_tif_path)   # shape: (Z, Y, X)

print(f"Image stack shape: {img_stack.shape}")
print(f"Mask stack shape:  {mask_stack.shape}")

# If grayscale, add channel dimension
if img_stack.ndim == 3:
    img_stack = np.expand_dims(img_stack, axis=-1)

num_slices = img_stack.shape[0]
h, w = img_stack.shape[1:3]
print(f"Found {num_slices} slices, each {h}x{w}")

# ---------------- Crop generation ---------------- #
crop_counter = 1
while crop_counter <= num_crops:
    z = random.randint(0, num_slices - 1)
    img_slice = img_stack[z]
    mask_slice = mask_stack[z]

    # random 2D crop within this slice
    y = random.randint(0, h - crop_size)
    x = random.randint(0, w - crop_size)

    img_crop = img_slice[y:y+crop_size, x:x+crop_size]
    mask_crop = mask_slice[y:y+crop_size, x:x+crop_size]

    # ensure 3 channels for image
    if img_crop.ndim == 2:
        img_crop = np.stack([img_crop]*3, axis=-1)

    # Save as uint8 tif
    imwrite(os.path.join(save_img_dir, f"img{crop_counter:03d}.tif"),
            img_crop.astype(np.uint8))
    imwrite(os.path.join(save_mask_dir, f"mask{crop_counter:03d}.tif"),
            mask_crop.astype(np.uint8))

    crop_counter += 1

print(f"✅ Saved {num_crops} 2D crops (size {crop_size}x{crop_size}) to:")
print(f"   {save_img_dir}")
print(f"   {save_mask_dir}")
