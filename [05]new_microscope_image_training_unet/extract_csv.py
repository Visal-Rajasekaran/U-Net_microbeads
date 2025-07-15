import os
import pandas as pd

# --- Paths ---
csv_path = "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\dataset\\grayscale_norm_selected\\all_centroids_v5.csv"
image_folder = "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\dataset\\grayscale_norm_selected_split\\more_than_300"
output_csv_path = "C:\\Users\\rsvis\\PycharmProjects\\U-Net_microbeads\\[05]new_microscope_image_training_unet\\dataset\\grayscale_norm_selected_split\\more_than_300\\dense_centroids.csv"

filename_column = "image_filename"  # Change if your column name differs

# --- Load CSV ---
df = pd.read_csv(csv_path)

# --- Get filenames in the folder ---
existing_files = set(os.listdir(image_folder))

# --- Filter rows where the image file exists ---
filtered_df = df[df[filename_column].isin(existing_files)]

print(f"Original rows: {len(df)}")
print(f"Filtered rows: {len(filtered_df)}")

# --- Save new CSV ---
filtered_df.to_csv(output_csv_path, index=False)

print(f"Saved filtered CSV to: {output_csv_path}")
