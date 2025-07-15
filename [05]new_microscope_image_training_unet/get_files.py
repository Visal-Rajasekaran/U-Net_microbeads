import os
import shutil

def copy_matching_files(dir1, dir2, output_dir, match_by_stem=True):
    """
    Copies files from dir1 to output_dir only if a file with the same name exists in dir2.

    Args:
        dir1 (str): Source directory to copy files from.
        dir2 (str): Directory to check for matching filenames.
        output_dir (str): Destination directory to save matching files.
        match_by_stem (bool): If True, match by filename without extension.
                              If False, match by full filename including extension.
    """
    os.makedirs(output_dir, exist_ok=True)

    files1 = os.listdir(dir1)
    files2 = os.listdir(dir2)

    if match_by_stem:
        names2 = {os.path.splitext(f)[0] for f in files2}
    else:
        names2 = set(files2)

    match_count = 0
    for f in files1:
        f_check = os.path.splitext(f)[0] if match_by_stem else f
        if f_check in names2:
            shutil.copy2(os.path.join(dir1, f), os.path.join(output_dir, f))
            match_count += 1

    print(f"[INFO] Files in {dir1}: {len(files1)}")
    print(f"[INFO] Files in {dir2}: {len(files2)}")
    print(f"[DONE] Copied {match_count} matching files to '{output_dir}'")

import os
import json
import shutil

def copy_non_matching_files(json_path, source_dir, output_dir):
    """
    Copies files from source_dir to output_dir if their names are NOT listed in the COCO JSON file.

    Args:
        json_path (str): Path to COCO-style JSON annotation file.
        source_dir (str): Directory containing candidate files.
        output_dir (str): Destination to save non-matching files.
    """
    # Load COCO JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract filenames from JSON
    json_filenames = {img['file_name'] for img in coco_data['images']}
    
    # Get all files in source directory
    source_filenames = os.listdir(source_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    non_matching = [f for f in source_filenames if f not in json_filenames]

    for f in non_matching:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(output_dir, f))

    print(f"[INFO] Files in COCO JSON: {len(json_filenames)}")
    print(f"[INFO] Files in source dir: {len(source_filenames)}")
    print(f"[DONE] Copied {len(non_matching)} non-matching files to '{output_dir}'")


import os
import json
import shutil

def copy_matching_files_from_coco(json_path, source_dir, output_dir):
    """
    Copies files from source_dir to output_dir if their filenames are listed in the COCO JSON file.

    Args:
        json_path (str): Path to COCO-style JSON annotation file.
        source_dir (str): Directory containing candidate files.
        output_dir (str): Destination to save matching files.
    """
    # Load COCO JSON
    with open(json_path, 'r') as f:
        coco_data = json.load(f)

    # Extract filenames from JSON
    json_filenames = {img['file_name'] for img in coco_data['images']}
    
    # Get all files in source directory
    source_filenames = os.listdir(source_dir)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Filter files that exist in both
    matching = [f for f in source_filenames if f in json_filenames]

    for f in matching:
        shutil.copy2(os.path.join(source_dir, f), os.path.join(output_dir, f))

    print(f"[INFO] Files listed in JSON: {len(json_filenames)}")
    print(f"[INFO] Files in source dir: {len(source_filenames)}")
    print(f"[DONE] Copied {len(matching)} matching files to '{output_dir}'")


from pathlib import Path
BASE_DIR = BASE_DIR = Path(__file__).resolve().parents[0]  # 1 levels up
INPUT_FOLDER = os.path.join(BASE_DIR, "dataset", "images")
#OUTPUT_FOLDER = os.path.join(BASE_DIR,"dataset", "fft_0.9")
#copy_matching_files(os.path.join(BASE_DIR, "dataset", "images"), os.path.join(BASE_DIR, "dataset", "fft_filtered_selected"), os.path.join(BASE_DIR, "dataset", "images_selected"))
copy_matching_files_from_coco(
    json_path=os.path.join(BASE_DIR, "label_generation", "annotations_final.json"),
    source_dir=os.path.join(BASE_DIR, "dataset", "images"),
    output_dir=os.path.join(BASE_DIR, "dataset", "images_selected")
)
