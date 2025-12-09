import numpy as np
import open3d as o3d
import pyarrow.parquet as pq
from sklearn.cluster import DBSCAN
import os

LIDAR_HEIGHT = 2.0
SHIFT_X = 1.6
SHIFT_Z = -3.0
DEFAULT_COLOR = np.array([0.6, 0.6, 0.6])

def read_lidar_points(parquet_path, target_timestamp=None):
    try:
        df = pq.read_table(parquet_path).to_pandas()
    except Exception as e:
        print(f"Failed reading LiDAR parquet: {e}")
        return None

    if df.empty:
        print("LiDAR parquet is empty.")
        return None

    target_timestamp = target_timestamp or df["key.frame_timestamp_micros"].iloc[0]
    df = df[df["key.frame_timestamp_micros"] == target_timestamp]

    if df.empty:
        print(f"No LiDAR data found for timestamp: {target_timestamp}")
        return None

    all_pts = []
    for idx in range(len(df)):
        try:
            x = df["[LiDARComponent].cartesian.x"].iloc[idx]
            y = df["[LiDARComponent].cartesian.y"].iloc[idx]
            z = df["[LiDARComponent].cartesian.z"].iloc[idx]
            pts = np.stack([x, y, z], axis=-1)
            all_pts.append(pts)
            continue
        except Exception:
            pass

        for rname in ["range_image_return1", "range_image_return2"]:
            vals = df[f"[LiDARComponent].{rname}.values"].iloc[idx]
            shape = df[f"[LiDARComponent].{rname}.shape"].iloc[idx]
            if vals is not None and shape is not None:
                xyz = process_range_image(vals, shape)
                if xyz is not None:
                    all_pts.append(xyz)

    if not all_pts:
        print("No LiDAR points found.")
        return None

    points = np.vstack(all_pts)
    points[:, 2] -= LIDAR_HEIGHT

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.paint_uniform_color(DEFAULT_COLOR)
    pcd = pcd.voxel_down_sample(0.05)

    print(f"Loaded LiDAR points: {len(pcd.points)}")
    return pcd

def spherical_to_xyz(range_image):
    H, W, _ = range_image.shape
    r = range_image[..., 0]
    v_angles = np.radians(np.linspace(5.0, -25.0, H))[:, None]
    a_angles = (np.arange(W) + 0.5) / W * 2 * np.pi
    a_angles = a_angles[None, :]
    x = r * np.cos(v_angles) * np.cos(a_angles)
    y = r * np.cos(v_angles) * np.sin(a_angles)
    z = r * np.sin(v_angles)
    mask = r > 0
    return np.stack([x, y, z], axis=-1)[mask]

def process_range_image(values, shape):
    try:
        arr = np.array(values).reshape(*shape)
        return spherical_to_xyz(arr)
    except Exception as e:
        print(f"Error in spherical decoding: {e}")
        return None

def create_box_lines(center, size, heading, color):
    l, w, h = size
    corners = np.array([
        [ l/2,  w/2,  h/2],
        [ l/2, -w/2,  h/2],
        [-l/2, -w/2,  h/2],
        [-l/2,  w/2,  h/2],
        [ l/2,  w/2, -h/2],
        [ l/2, -w/2, -h/2],
        [-l/2, -w/2, -h/2],
        [-l/2,  w/2, -h/2],
    ])
    R = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading),  np.cos(heading), 0],
        [0, 0, 1]
    ])
    corners = (R @ corners.T).T + center
    lines = [
        [0,1],[1,2],[2,3],[3,0],
        [4,5],[5,6],[6,7],[7,4],
        [0,4],[1,5],[2,6],[3,7]
    ]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(corners),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([color]*len(lines))
    return line_set

def get_box_obb(center, size, heading):
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle(
        np.array([0, 0, heading])
    )
    return o3d.geometry.OrientedBoundingBox(center, R, size)

def read_lidar_boxes(parquet_path, target_timestamp=None):
    TYPE_COLOR = {1:[1,0,0], 2:[0,1,0], 3:[0,0,1], 4:[1,1,0]}
    try:
        df = pq.read_table(parquet_path).to_pandas()
    except Exception as e:
        print(f"Failed reading box parquet: {e}")
        return [], [], [], []

    if df.empty:
        return [], [], [], []

    target_timestamp = target_timestamp or df["key.frame_timestamp_micros"].iloc[0]
    df = df[df["key.frame_timestamp_micros"] == target_timestamp]

    boxes, obbs, colors, types = [], [], [], []

    for _, row in df.iterrows():
        c = np.array([
            -row["[LiDARBoxComponent].box.center.x"] + SHIFT_X,
            row["[LiDARBoxComponent].box.center.y"],
            row["[LiDARBoxComponent].box.center.z"] - LIDAR_HEIGHT + SHIFT_Z
        ])
        size = np.array([
            row["[LiDARBoxComponent].box.size.x"],
            row["[LiDARBoxComponent].box.size.y"],
            row["[LiDARBoxComponent].box.size.z"]
        ])
        heading = row["[LiDARBoxComponent].box.heading"] + np.pi
        t = row["[LiDARBoxComponent].type"]
        color = TYPE_COLOR.get(t, [1,1,1])
        boxes.append(create_box_lines(c, size, heading, color))
        obbs.append(get_box_obb(c, size, heading))
        colors.append(color)
        types.append(t)

    print(f"Loaded {len(boxes)} bounding boxes.")
    return boxes, obbs, colors, types

def assign_box_colors(pcd, obbs, box_colors):
    points = np.asarray(pcd.points)
    final_colors = np.tile(DEFAULT_COLOR, (len(points),1))
    clustering = DBSCAN(eps=0.3, min_samples=5).fit(points)
    labels = clustering.labels_

    for obb, color in zip(obbs, box_colors):
        inside_ids = obb.get_point_indices_within_bounding_box(pcd.points)
        if len(inside_ids) == 0:
            continue
        touched_clusters = set(labels[inside_ids])
        touched_clusters.discard(-1)
        for c in touched_clusters:
            final_colors[labels == c] = color
        final_colors[inside_ids] = color

    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    return pcd

def extract_top_vehicles(pcd, obbs, box_colors, box_types, top_n=5, min_points=50):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    vehicles = []

    for obb, color, t in zip(obbs, box_colors, box_types):
        if t != 1:
            continue
        inside_idx = obb.get_point_indices_within_bounding_box(pcd.points)
        if len(inside_idx) < min_points:
            continue
        veh_pcd = o3d.geometry.PointCloud()
        veh_pcd.points = o3d.utility.Vector3dVector(points[inside_idx])
        veh_pcd.colors = o3d.utility.Vector3dVector(colors[inside_idx])
        vehicles.append((veh_pcd, obb, color, len(inside_idx)))

    vehicles.sort(key=lambda x: x[3], reverse=True)
    top_vehicles = vehicles[:top_n]
    print(f"Selected top {len(top_vehicles)} vehicle point clouds.")
    return top_vehicles

def save_top_vehicle_point_clouds(vehicle_clouds, save_dir="vehicles_out"):
    os.makedirs(save_dir, exist_ok=True)
    for i, (veh_pcd, _, _, _) in enumerate(vehicle_clouds):
        path = f"{save_dir}/vehicle_{i+1:02d}.ply"
        o3d.io.write_point_cloud(path, veh_pcd)
        print(f"Saved {path}")

def visualize_individual_vehicle(vehicle_clouds):
    for i, (veh_pcd, obb, color, _) in enumerate(vehicle_clouds):
        print(f"Displaying VEHICLE #{i+1}, points={len(veh_pcd.points)}")
        lineset = create_box_lines(obb.center, obb.extent, 0, color)
        o3d.visualization.draw_geometries([veh_pcd, lineset])

def visualize(pcd, boxes):
    geoms = [pcd] + boxes + [o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)]
    o3d.visualization.draw_geometries(geoms)

def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python script.py <lidar.parquet> <boxes.parquet> [timestamp]")
        return

    lidar_path, box_path = sys.argv[1], sys.argv[2]
    timestamp = int(sys.argv[3]) if len(sys.argv) > 3 else None

    pcd = read_lidar_points(lidar_path, timestamp)
    boxes, obbs, colors, types = read_lidar_boxes(box_path, timestamp)

    if pcd:
        pcd = assign_box_colors(pcd, obbs, colors)

    top_vehicles = extract_top_vehicles(pcd, obbs, colors, types, top_n=5, min_points=50)
    save_top_vehicle_point_clouds(top_vehicles)
    visualize_individual_vehicle(top_vehicles) 

    visualize(pcd, boxes)

if __name__ == "__main__":
    main()
