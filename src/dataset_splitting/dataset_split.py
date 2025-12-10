import os
import shutil
import random

INPUT_FOLDER = "waymo_preprocessing/waymo_vehicle_ground_truth_occluded_pointclouds/extracted_50_min_points_updated_occluded_gt"
OUTPUT_FOLDER = "waymo_final_data_splits_updated"
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}
SUBFOLDERS = ["incomplete", "completed"]

def create_splits(input_folder, output_folder, splits, subfolders):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for split_name in splits:
        for subfolder in subfolders:
            os.makedirs(os.path.join(output_folder, split_name, subfolder), exist_ok=True)

    incomplete_folder = os.path.join(input_folder, "incomplete")
    completed_folder = os.path.join(input_folder, "completed")

    incomplete_files = [f for f in os.listdir(incomplete_folder) if os.path.isfile(os.path.join(incomplete_folder, f))]
    incomplete_files.sort()  
    random.shuffle(incomplete_files)

    n = len(incomplete_files)
    n_train = int(n * splits["train"])
    n_val = int(n * splits["val"])
    n_test = n - n_train - n_val

    split_files = {
        "train": incomplete_files[:n_train],
        "val": incomplete_files[n_train:n_train+n_val],
        "test": incomplete_files[n_train+n_val:]
    }

    for split_name, files in split_files.items():
        for f in files:
            src_inc = os.path.join(incomplete_folder, f)
            dst_inc = os.path.join(output_folder, split_name, "incomplete", f)
            shutil.copy2(src_inc, dst_inc)

            src_comp = os.path.join(completed_folder, f)
            dst_comp = os.path.join(output_folder, split_name, "completed", f)
            if os.path.exists(src_comp):
                shutil.copy2(src_comp, dst_comp)
            else:
                print(f"Warning: completed file missing for {f}")

    print("Dataset split completed!")

if __name__ == "__main__":
    create_splits(INPUT_FOLDER, OUTPUT_FOLDER, SPLITS, SUBFOLDERS)
