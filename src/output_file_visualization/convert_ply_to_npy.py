import os
import numpy as np
import open3d as o3d

FILE_PATH = None
FOLDER_PATH = "waymo_final_data_splits_updated_normalized/val/completed"   

def convert_single_ply(ply_path, npy_path):
    try:
        pc = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pc.points)

        if points.size == 0:
            print(f"Warning: {ply_path} contains no points.")

        np.save(npy_path, points)
        print(f"Converted: {ply_path} -> {npy_path}")

    except Exception as e:
        print(f"Error converting {ply_path}: {e}")

def convert_folder(input_folder):
    output_folder = input_folder.rstrip("/").rstrip("\\") + "_npy"
    os.makedirs(output_folder, exist_ok=True)

    print(f"\nSaving converted files to: {output_folder}\n")

    for root, dirs, files in os.walk(input_folder):
        rel_path = os.path.relpath(root, input_folder)
        save_dir = os.path.join(output_folder, rel_path)
        os.makedirs(save_dir, exist_ok=True)

        for f in files:
            if f.lower().endswith(".ply"):
                ply_path = os.path.join(root, f)
                npy_path = os.path.join(save_dir, f.replace(".ply", ".npy"))
                convert_single_ply(ply_path, npy_path)

    print("\nConversion complete.")

if __name__ == "__main__":
    if FILE_PATH and os.path.isfile(FILE_PATH):
        output_path = FILE_PATH.replace(".ply", ".npy")
        convert_single_ply(FILE_PATH, output_path)
    elif FOLDER_PATH and os.path.isdir(FOLDER_PATH):
        convert_folder(FOLDER_PATH)
    else:
        print("Invalid path.")
