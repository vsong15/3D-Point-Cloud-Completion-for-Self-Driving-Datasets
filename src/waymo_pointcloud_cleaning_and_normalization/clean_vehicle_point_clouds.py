import os
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN

FILE_PATH = None
FOLDER_PATH = "waymo_vehicle_incomplete_pointclouds/all_vehicle_point_clouds_1000_min_points"
OUTPUT_FOLDER = "waymo_vehicle_cleaned_pointclouds/all_vehicle_point_clouds_1000_min_points_cleaned"
MAX_DIAGONAL = 4.0
PLANAR_THRESHOLD = 0.01  

def clean_points(pc):
    pc, _ = pc.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    pc = pc.voxel_down_sample(voxel_size=0.05)
    return pc

def has_single_cluster(pc, eps=0.5, min_samples=10):
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return False
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pts)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters == 1

def is_compact(pc, max_diagonal=MAX_DIAGONAL):
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return False
    bbox = pts.max(axis=0) - pts.min(axis=0)
    diag = np.linalg.norm(bbox)
    return diag <= max_diagonal

def is_planar(pc, threshold=PLANAR_THRESHOLD):
    pts = np.asarray(pc.points)
    if len(pts) < 3:
        return True
    cov = np.cov(pts.T)
    eigvals = np.linalg.eigvalsh(cov)
    return eigvals[0] / eigvals[2] < threshold  

def process_file(path):
    pc = o3d.io.read_point_cloud(path)
    pc = clean_points(pc)
    if not has_single_cluster(pc):
        return None
    if not is_compact(pc):
        return None
    if is_planar(pc):
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
