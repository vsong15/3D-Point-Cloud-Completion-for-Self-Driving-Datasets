import numpy as np
import open3d as o3d
import pyarrow.parquet as pq
import os

def spherical_to_xyz(range_image):
    H, W, C = range_image.shape
    ranges = range_image[..., 0]
    vertical = np.linspace(5.0, -25.0, H) * np.pi / 180.0
    vertical = vertical[:, None]
    azimuth = (np.arange(W) + 0.5) / W * 2 * np.pi
    azimuth = azimuth[None, :]
    x = ranges * np.cos(vertical) * np.cos(azimuth)
    y = ranges * np.cos(vertical) * np.sin(azimuth)
    z = ranges * np.sin(vertical)
    mask = ranges > 0
    pts = np.stack([x, y, z], axis=-1)
    return pts[mask]

def process_range_image(values, shape):
    try:
        pts = np.array(values).reshape(shape[0], shape[1], shape[2])
        print(f"Range image shape: {pts.shape}")
        xyz = spherical_to_xyz(pts)
        print(f"Converted {len(xyz)} valid points")
        return xyz
    except Exception as e:
        print(f"Error processing range image: {e}")
        return None

def read_and_process_lidar(parquet_path, target_timestamp=None):
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        print("\nPoint parquet columns:")
        for col in df.columns:
            print(f"- {col}")

        timestamps = df["key.frame_timestamp_micros"].unique()
        print(f"\nFound {len(timestamps)} timestamps:")
        for ts in timestamps:
            print(f"Timestamp: {ts}")

        if target_timestamp is None:
            target_timestamp = timestamps[0]
            print(f"\nUsing first timestamp: {target_timestamp}")

        df = df[df["key.frame_timestamp_micros"] == target_timestamp]
        print(f"\nProcessing data for timestamp: {target_timestamp}")
        print(f"Number of laser returns: {len(df)}")

        all_xyz = []

        for idx in range(len(df)):
            print(f"\nProcessing laser index {idx}...")
            vals1 = df["[LiDARComponent].range_image_return1.values"].iloc[idx]
            shape1 = df["[LiDARComponent].range_image_return1.shape"].iloc[idx]
            print(f"  return1 shape: {shape1}, values length: {len(vals1)}")
            print("Processing return1 range image...")
            xyz1 = process_range_image(vals1, shape1)
            if xyz1 is not None:
                all_xyz.append(xyz1)

            vals2 = df["[LiDARComponent].range_image_return2.values"].iloc[idx]
            if len(vals2) > 0:
                shape2 = df["[LiDARComponent].range_image_return2.shape"].iloc[idx]
                print(f"  return2 shape: {shape2}, values length: {len(vals2)}")
                print("Processing return2 range image...")
                xyz2 = process_range_image(vals2, shape2)
                if xyz2 is not None:
                    all_xyz.append(xyz2)

        if len(all_xyz) > 0:
            pts = np.vstack(all_xyz)
            print(f"\nTotal points for timestamp {target_timestamp}: {len(pts)}")
            return pts

        return None
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        return None

def visualize_points(points):
    if points is None or len(points) == 0:
        print("No points to visualize.")
        return
    print(f"\nVisualizing {len(points)} points...")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python dataset_read.py <file.parquet> [timestamp]")
        return
    parquet_path = sys.argv[1]
    timestamp = int(sys.argv[2]) if len(sys.argv) > 2 else None
    points = read_and_process_lidar(parquet_path, timestamp)
    visualize_points(points)

if __name__ == "__main__":
    main()
