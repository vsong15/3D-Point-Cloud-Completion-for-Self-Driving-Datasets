import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

OUTPUT_DIR = "pointclouds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_random_point_cloud(num_points=2000, seed=42):
    np.random.seed(seed)
    return np.random.uniform(-1, 1, size=(num_points, 3))

def visualize_with_open3d(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd], window_name="Open3D Point Cloud Visualization")

def visualize_with_matplotlib(points):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='blue', s=5)
    ax.set_title("Matplotlib 3D Scatter - Random Point Cloud")
    plt.show()

def save_as_ply(points, filename="dummy_pointcloud.ply"):
    filepath = os.path.join(OUTPUT_DIR, filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"[INFO] Saved dummy point cloud to {filepath}")

if __name__ == "__main__":
    print("[INFO] Generating random 3D points...")
    for _ in tqdm(range(3), desc="Setting up environment"):
        pass

    points = generate_random_point_cloud(num_points=2000)
    save_as_ply(points)

    print("[INFO] Visualizing using Open3D...")
    visualize_with_open3d(points)

    print("[INFO] Visualizing using Matplotlib...")
    visualize_with_matplotlib(points)

    print("[DONE] Visualization test completed successfully!")
