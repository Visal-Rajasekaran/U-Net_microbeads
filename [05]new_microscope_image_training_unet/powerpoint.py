import os
import shutil

# --- Paths ---
folder_A =  "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\dataset\\masks"
folder_B =  "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\dataset\\grayscale_norm_selected_split\\more_than_300"
output_folder =  "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\dataset\\grayscale_norm_selected_masks_dense"

os.makedirs(output_folder, exist_ok=True)

# --- List files ---
files_A = set(os.listdir(folder_A))
files_B = set(os.listdir(folder_B))

# --- Find common filenames ---
common_files = files_A & files_B  # intersection

print(f"Found {len(common_files)} common files.")

# --- Copy from one source (e.g., folder_A) to output folder ---
for filename in common_files:
    src_path = os.path.join(folder_A, filename)
    dst_path = os.path.join(output_folder, filename)
    shutil.copy2(src_path, dst_path)

print("Done copying common files.")
