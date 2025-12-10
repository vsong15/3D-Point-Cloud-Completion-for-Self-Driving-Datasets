import os
import numpy as np
import open3d as o3d

ROOT_FOLDER = "waymo_final_data_splits_updated"
OUTPUT_FOLDER = "waymo_final_data_splits_updated_normalized"

def normalize_unit_sphere(pc):
    """Normalize a point cloud so all points lie in [-1, 1] in XYZ."""
    pts = np.asarray(pc.points)

    centroid = pts.mean(axis=0)
    pts = pts - centroid

    max_dist = np.linalg.norm(pts, axis=1).max()
    if max_dist > 0:
        pts = pts / max_dist

    pc.points = o3d.utility.Vector3dVector(pts)
    return pc

def process_ply_file(input_path, output_path):
    pc = o3d.io.read_point_cloud(input_path)
    if pc.is_empty():
        print("Skipping empty file:", input_path)
        return

    pc = normalize_unit_sphere(pc)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    o3d.io.write_point_cloud(output_path, pc)
    print("Saved:", output_path)

def process_dataset(root):
    for split in ["train", "val", "test"]:
        for state in ["completed", "incomplete"]:
            folder = os.path.join(root, split, state)
            if not os.path.exists(folder):
                print("Missing folder:", folder)
                continue

            for file in sorted(os.listdir(folder)):
                if file.lower().endswith(".ply"):
                    input_path = os.path.join(folder, file)
                    
                    output_path = os.path.join(
                        OUTPUT_FOLDER, split, state, file
                    )

                    process_ply_file(input_path, output_path)

if __name__ == "__main__":
    process_dataset(ROOT_FOLDER)
    print("\nFinished normalization into:", OUTPUT_FOLDER)
