import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

FILE_PATH = None
FOLDER_PATH = "waymo_preprocessing/waymo_vehicle_extracted_pointclouds/extracted_50_min_points_updated"
OUTPUT_FOLDER = "waymo_preprocessing/waymo_vehicle_cleaned_pointclouds/extracted_50_min_points_updated_cleaned"
MAX_DIAGONAL = 6.0

def clean_points(pc):
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=10, std_ratio=3.0)
    return pc

def keep_largest_cluster(pc, eps=0.4, min_samples=5):
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return pc
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = clustering.labels_
    unique = np.unique(labels[labels >= 0])
    if len(unique) == 0:
        return pc
    largest_label = max(unique, key=lambda lbl: np.sum(labels == lbl))
    mask = labels == largest_label
    filtered_pts = pts[mask]
    new_pc = o3d.geometry.PointCloud()
    new_pc.points = o3d.utility.Vector3dVector(filtered_pts)
    return new_pc

def is_compact(pc, max_diagonal=MAX_DIAGONAL):
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return False
    bbox = pts.max(axis=0) - pts.min(axis=0)
    diag = np.linalg.norm(bbox)
    return diag <= max_diagonal

def process_file(path):
    pc = o3d.io.read_point_cloud(path)
    pc = clean_points(pc)
    pc = keep_largest_cluster(pc)
    if not is_compact(pc):
        return None
    pc.paint_uniform_color([1.0, 0.0, 0.0])
    return pc

def process_folder(root):
    filtered_pcs = []
    for r, d, files in os.walk(root):
        for name in sorted(files):
            if name.lower().endswith(".ply"):
                path = os.path.join(r, name)
                pc = process_file(path)
                if pc is not None:
                    filtered_pcs.append(pc)
    return filtered_pcs

def save_pointclouds(pcs):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for i, pc in enumerate(pcs, start=1):
        out_path = os.path.join(OUTPUT_FOLDER, f"vehicle_{i:04d}.ply")
        o3d.io.write_point_cloud(out_path, pc)
        print("Saved", out_path)

if __name__ == "__main__":
    pcs = []
    if FILE_PATH and os.path.isfile(FILE_PATH):
        pc = process_file(FILE_PATH)
        if pc is not None:
            pcs = [pc]
    elif FOLDER_PATH and os.path.isdir(FOLDER_PATH):
        pcs = process_folder(FOLDER_PATH)
    else:
        print("Invalid path.")
    if pcs:
        save_pointclouds(pcs)
