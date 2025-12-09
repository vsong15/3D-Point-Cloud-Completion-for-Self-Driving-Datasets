import os
import numpy as np
import open3d as o3d

FILE_PATH = None
FOLDER_PATH = "inference_result"

def visualize_array(arr, title):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(arr[:, :3])
    o3d.visualization.draw_geometries([pc], window_name=title)

def visualize_single(file_path):
    arr = np.load(file_path)
    visualize_array(arr, os.path.basename(file_path))

def visualize_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".npy")]
    for f in files:
        visualize_single(os.path.join(folder_path, f))

def visualize_nested(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for f in files:
            if f.lower().endswith(".npy"):
                visualize_single(os.path.join(root, f))

if __name__ == "__main__":
    if FILE_PATH and os.path.isfile(FILE_PATH):
        visualize_single(FILE_PATH)
    elif FOLDER_PATH and os.path.isdir(FOLDER_PATH):
        visualize_nested(FOLDER_PATH)
    else:
        print("Invalid path.")
