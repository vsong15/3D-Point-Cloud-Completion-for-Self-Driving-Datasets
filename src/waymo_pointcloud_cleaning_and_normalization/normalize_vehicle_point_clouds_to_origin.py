import os
import numpy as np
import open3d as o3d

INPUT_FOLDER = "waymo_preprocessing/waymo_vehicle_cleaned_pointclouds/extracted_50_min_points_updated_cleaned"
OUTPUT_FOLDER = "waymo_preprocessing/waymo_vehicle_normalized_pointclouds/extracted_50_min_points_updated_normalized"

MIN_POINTS = 1000
MAX_DIAGONAL = 6.0

def is_valid(pc, min_points=MIN_POINTS, max_diagonal=MAX_DIAGONAL):
    pts = np.asarray(pc.points)
    if len(pts) < min_points:
        return False
    bbox = pts.max(axis=0) - pts.min(axis=0)
    diag = np.linalg.norm(bbox)
    return diag <= max_diagonal

def normalize_to_origin(pc):
    pts = np.asarray(pc.points)
    min_corner = pts.min(axis=0)
    pts -= min_corner
    diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    if diag > 0:
        pts /= diag
    pc.points = o3d.utility.Vector3dVector(pts)
    pc.paint_uniform_color([1.0, 0.0, 0.0])
    return pc

def process_file(path):
    pc = o3d.io.read_point_cloud(path)
    if pc.is_empty():
        return None
    if not is_valid(pc):
        return None
    pc = normalize_to_origin(pc)
    return pc

def process_folder(root):
    normalized_pcs = []
    for r, d, files in os.walk(root):
        for name in sorted(files):
            if name.lower().endswith(".ply"):
                path = os.path.join(r, name)
                pc = process_file(path)
                if pc is not None:
                    normalized_pcs.append(pc)
    return normalized_pcs

def save_pointclouds(pcs):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for i, pc in enumerate(pcs, start=1):
        out_path = os.path.join(OUTPUT_FOLDER, f"vehicle_{i:04d}.ply")
        o3d.io.write_point_cloud(out_path, pc)
        print("Saved", out_path)

if __name__ == "__main__":
    pcs = process_folder(INPUT_FOLDER)
    if pcs:
        save_pointclouds(pcs)
    else:
        print("No valid point clouds found.")
