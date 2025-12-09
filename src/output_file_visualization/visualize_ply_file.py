import os
import open3d as o3d

FILE_PATH = None
FOLDER_PATH = "waymo_vehicle_ground_truth_occluded/all_vehicle_point_clouds_50_min_points_occluded_gt/incomplete"

def visualize_single(file_path):
    pc = o3d.io.read_point_cloud(file_path)
    if pc.is_empty():
        print(f"Warning: {file_path} is empty or unreadable.")
        return
    o3d.visualization.draw_geometries([pc], window_name=os.path.basename(file_path))

def visualize_folder(folder_path):
    ply_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".ply")]
    if not ply_files:
        print("No .ply files found.")
        return
    for ply_file in ply_files:
        visualize_single(os.path.join(folder_path, ply_file))

if __name__ == "__main__":
    if FILE_PATH and os.path.isfile(FILE_PATH):
        visualize_single(FILE_PATH)
    elif FOLDER_PATH and os.path.isdir(FOLDER_PATH):
        visualize_folder(FOLDER_PATH)
    else:
        print("Set FILE_PATH to a .ply file or FOLDER_PATH to a directory.")
