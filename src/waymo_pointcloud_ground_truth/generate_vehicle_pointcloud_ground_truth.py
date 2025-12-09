import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

INPUT_FOLDER = "waymo_vehicle_normalized_pointclouds/all_vehicle_point_clouds_1000_min_points_normalized"
GROUND_TRUTH_FOLDER = "waymo_vehicle_ground_truth_occluded/all_vehicle_point_clouds_1000_min_points_occluded_gt"
INCOMPLETE_FOLDER = os.path.join(GROUND_TRUTH_FOLDER, "incomplete").replace("\\", "/")
COMPLETED_FOLDER = os.path.join(GROUND_TRUTH_FOLDER, "completed").replace("\\", "/")

OCCLUDER_SIZE = np.array([5.5, 9.0, 6.5])
OCCLUDER_DISTANCE = 0
MIN_OCCLUDED_POINTS = 50 

def apply_occlusion(pc):
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return None

    min_bounds = pts.min(axis=0)
    max_bounds = pts.max(axis=0)
    center = (min_bounds + max_bounds) / 2
    vehicle_size = max_bounds - min_bounds
    scaled_occluder = np.minimum(OCCLUDER_SIZE, vehicle_size * 0.8)
    occ_min = center + np.array([-scaled_occluder[0]/2, OCCLUDER_DISTANCE, -scaled_occluder[2]/2])
    occ_max = occ_min + scaled_occluder

    mask_x = (pts[:,0] < occ_min[0]) | (pts[:,0] > occ_max[0])
    mask_y = (pts[:,1] < occ_min[1]) | (pts[:,1] > occ_max[1])
    mask_z = (pts[:,2] < occ_min[2]) | (pts[:,2] > occ_max[2])
    mask = mask_x | mask_y | mask_z

    num_occluded = len(pts) - np.sum(mask)
    if num_occluded < MIN_OCCLUDED_POINTS:
        return None  

    occluded_pts = pts[mask]
    pc_occluded = o3d.geometry.PointCloud()
    pc_occluded.points = o3d.utility.Vector3dVector(occluded_pts)
    pc_occluded.paint_uniform_color([1.0, 0.0, 0.0])

    return pc_occluded

def process_folder(root):
    os.makedirs(INCOMPLETE_FOLDER, exist_ok=True)
    os.makedirs(COMPLETED_FOLDER, exist_ok=True)

    ply_files = sorted([f for f in os.listdir(root) if f.lower().endswith(".ply")])
    idx = 1
    for name in ply_files:
        path = os.path.join(root, name)
        pc = o3d.io.read_point_cloud(path)
        if pc.is_empty():
            continue

        occluded_pc = apply_occlusion(pc)
        if occluded_pc is None:
            print(f"Warning: Occluder missed all points in {name}, skipping file.")
            continue

        completed_out = os.path.join(COMPLETED_FOLDER, f"vehicle_{idx:04d}.ply")
        o3d.io.write_point_cloud(completed_out, pc)

        incomplete_out = os.path.join(INCOMPLETE_FOLDER, f"vehicle_{idx:04d}.ply")
        o3d.io.write_point_cloud(incomplete_out, occluded_pc)

        print(f"Saved GT {idx:04d}: incomplete + completed")
        idx += 1

if __name__ == "__main__":
    process_folder(INPUT_FOLDER)
    print("Ground truth dataset with occlusions created successfully!")
