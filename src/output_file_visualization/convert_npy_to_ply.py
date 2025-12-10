import os
import numpy as np
import open3d as o3d

PRED_FOLDER = "inference_result_non_fine_tuned_incomplete"
OUTPUT_FOLDER = "inference_result_non_fine_tuned_incomplete_ply"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for sub in os.listdir(PRED_FOLDER):
    sub_path = os.path.join(PRED_FOLDER, sub)
    if not os.path.isdir(sub_path):
        continue
    npy_file = os.path.join(sub_path, "fine.npy")
    if not os.path.isfile(npy_file):
        print(f"Warning: {sub} missing fine.npy → skipped")
        continue

    points = np.load(npy_file).astype(np.float32)
    if points.size == 0:
        print(f"Warning: {sub} fine.npy is empty → skipped")
        continue

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    ply_file = os.path.join(OUTPUT_FOLDER, f"{sub}.ply")
    o3d.io.write_point_cloud(ply_file, pc)
    print(f"Converted: {npy_file} -> {ply_file}")
