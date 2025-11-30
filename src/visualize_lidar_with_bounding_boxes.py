import numpy as np
import open3d as o3d
import pyarrow.parquet as pq

LIDAR_HEIGHT = 2.0  

def read_lidar_points(parquet_path, target_timestamp=None):
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        if df.empty:
            print("LiDAR parquet file is empty!")
            return None

        if target_timestamp is None:
            target_timestamp = df["key.frame_timestamp_micros"].iloc[0]

        df = df[df["key.frame_timestamp_micros"] == target_timestamp]
        if df.empty:
            print(f"No LiDAR data found for timestamp {target_timestamp}")
            return None

        all_points = []

        for idx in range(len(df)):
            try:
                x = df[f"[LiDARComponent].cartesian.x"].iloc[idx]
                y = df[f"[LiDARComponent].cartesian.y"].iloc[idx]
                z = df[f"[LiDARComponent].cartesian.z"].iloc[idx]

                pts = np.stack([x, y, z], axis=-1)
                all_points.append(pts)
            except:
                for return_name in ["range_image_return1", "range_image_return2"]:
                    vals = df[f"[LiDARComponent].{return_name}.values"].iloc[idx]
                    shape = df[f"[LiDARComponent].{return_name}.shape"].iloc[idx]
                    if vals is not None and shape is not None:
                        xyz = process_range_image(vals, shape)
                        if xyz is not None and len(xyz) > 0:
                            all_points.append(xyz)

        if not all_points:
            print("No points found in this frame.")
            return None

        points = np.vstack(all_points)
        points[:, 2] -= LIDAR_HEIGHT

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.7, 0.7, 0.7])
        pcd = pcd.voxel_down_sample(0.05)
        print(f"Total LiDAR points loaded: {len(pcd.points)}")
        return pcd

    except Exception as e:
        print(f"Error reading LiDAR parquet file: {e}")
        return None

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
        xyz = spherical_to_xyz(pts)
        return xyz
    except Exception as e:
        print(f"Error processing range image: {e}")
        return None

def create_box_lines(center, size, heading=0.0, color=[1,0,0]):
    l, w, h = size
    corners = np.array([
        [l/2, w/2, h/2],
        [l/2, -w/2, h/2],
        [-l/2, -w/2, h/2],
        [-l/2, w/2, h/2],
        [l/2, w/2, -h/2],
        [l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2, w/2, -h/2],
    ])
    R = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading), np.cos(heading), 0],
        [0, 0, 1]
    ])
    corners = (R @ corners.T).T + np.array(center)
    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    colors = [color for _ in lines]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set, corners

def read_lidar_boxes(parquet_path, target_timestamp=None):
    TYPE_COLOR = {
        1: [1,0,0],    # Vehicle
        2: [0,1,0],    # Pedestrian
        3: [0,0,1],    # Cyclist
        4: [1,1,0],    # Sign/Other
    }
    try:
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        if df.empty:
            print("Boxes parquet file is empty!")
            return []

        if target_timestamp is None:
            target_timestamp = df["key.frame_timestamp_micros"].iloc[0]

        df = df[df["key.frame_timestamp_micros"] == target_timestamp]

        boxes = []
        for _, row in df.iterrows():
            obj_type = row["[LiDARBoxComponent].type"]
            color = TYPE_COLOR.get(obj_type, [1,1,1])

            center_x = -row["[LiDARBoxComponent].box.center.x"]
            center_z = row["[LiDARBoxComponent].box.center.z"] - LIDAR_HEIGHT

            SHIFT_X = 1.6
            SHIFT_Z = -3.0
            center = [
                center_x + SHIFT_X,
                row["[LiDARBoxComponent].box.center.y"],
                center_z + SHIFT_Z  
            ]

            heading = row["[LiDARBoxComponent].box.heading"] + np.pi

            size = [
                row["[LiDARBoxComponent].box.size.x"],
                row["[LiDARBoxComponent].box.size.y"],
                row["[LiDARBoxComponent].box.size.z"]
            ]

            line_set, _ = create_box_lines(center, size, heading, color)
            boxes.append(line_set)

        print(f"Total boxes loaded: {len(boxes)}")
        return boxes
    except Exception as e:
        print(f"Error reading LiDAR boxes: {e}")
        return []

def visualize(pcd, boxes):
    geometries = []
    if pcd is not None and len(pcd.points) > 0:
        geometries.append(pcd)
    geometries.extend(boxes)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    geometries.append(axes)
    o3d.visualization.draw_geometries(geometries)

def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python visualize_lidar_with_bounding_boxes.py <lidar.parquet> <boxes.parquet> [timestamp]")
        return

    lidar_path = sys.argv[1]
    boxes_path = sys.argv[2]
    timestamp = int(sys.argv[3]) if len(sys.argv) > 3 else None

    pcd = read_lidar_points(lidar_path, timestamp)
    boxes = read_lidar_boxes(boxes_path, timestamp)
    visualize(pcd, boxes)

if __name__ == "__main__":
    main()
