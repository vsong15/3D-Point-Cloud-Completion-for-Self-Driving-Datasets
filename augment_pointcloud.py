import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os

OUTPUT_DIR = "pointclouds"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_point_cloud(file_path=None, num_points=2000, seed=42):
    np.random.seed(seed)
    if file_path is None:
        print("[INFO] No file provided — generating synthetic point cloud.")
        return np.random.uniform(-1, 1, size=(num_points, 3))

    ext = os.path.splitext(file_path)[1]
    if ext == ".ply":
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points)
    elif ext == ".npy":
        return np.load(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def save_point_cloud(points, filename="augmented_pointcloud.ply"):
    filepath = os.path.join(OUTPUT_DIR, filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filepath, pcd)
    print(f"[INFO] Saved augmented point cloud to {filepath}")

def drop_points(points, drop_ratio=0.3):
    n = len(points)
    keep = int(n * (1 - drop_ratio))
    idx = np.random.choice(n, keep, replace=False)
    return points[idx]

def add_noise(points, sigma=0.02):
    return points + np.random.normal(0, sigma, size=points.shape)

def visualize_before_after(original, augmented, title1="Original", title2="Augmented"):
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(original[:, 0], original[:, 1], original[:, 2], c='blue', s=5)
    ax1.set_title(title1)
    ax1.axis('off')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(augmented[:, 0], augmented[:, 1], augmented[:, 2], c='red', s=5)
    ax2.set_title(title2)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    points = load_point_cloud("pointclouds/dummy_pointcloud.ply" 
                              if os.path.exists("pointclouds/dummy_pointcloud.ply") else None)

    print("[INFO] Applying point dropping...")
    dropped_points = drop_points(points, drop_ratio=0.4)
    save_point_cloud(dropped_points, "dropped_pointcloud.ply")

    print("[INFO] Applying Gaussian noise...")
    noisy_points = add_noise(points, sigma=0.02)
    save_point_cloud(noisy_points, "noisy_pointcloud.ply")

    visualize_before_after(points, dropped_points, "Clean Point Cloud", "Occluded (Dropped) Points")
    visualize_before_after(points, noisy_points, "Clean Point Cloud", "Noisy Points")

    print("[DONE] Augmentation visualization complete.")
