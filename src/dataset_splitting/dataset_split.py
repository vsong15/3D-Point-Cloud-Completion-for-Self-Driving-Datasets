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

    for subfolder in subfolders:
        src_folder = os.path.join(input_folder, subfolder)
        files = [f for f in os.listdir(src_folder) if os.path.isfile(os.path.join(src_folder, f))]
        random.shuffle(files)
        
        n = len(files)
        n_train = int(n * splits["train"])
        n_val = int(n * splits["val"])
        n_test = n - n_train - n_val

        split_files = {
            "train": files[:n_train],
            "val": files[n_train:n_train+n_val],
            "test": files[n_train+n_val:]
        }

        for split_name, split_list in split_files.items():
            for f in split_list:
                src_path = os.path.join(src_folder, f)
                dst_path = os.path.join(output_folder, split_name, subfolder, f)
                shutil.copy2(src_path, dst_path)

    print("Dataset split completed!")

if __name__ == "__main__":
    create_splits(INPUT_FOLDER, OUTPUT_FOLDER, SPLITS, SUBFOLDERS)
