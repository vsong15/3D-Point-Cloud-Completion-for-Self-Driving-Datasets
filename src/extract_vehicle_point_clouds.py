import os
import re
import open3d as o3d
import pyarrow.parquet as pq
import numpy as np
from sklearn.cluster import DBSCAN

LIDAR_DIR = "data/training/254_training_lidar"
BOX_DIR    = "data/training/254_training_lidar_box"
OUTPUT_DIR = "vehicles_out_200_min_points"
ALL_VEH_DIR = "all_vehicle_point_clouds_200_min_points"

LIDAR_HEIGHT = 2.0
SHIFT_X = 1.6
SHIFT_Z = -3.0
DEFAULT_COLOR = np.array([0.6, 0.6, 0.6])

def extract_id(fname):
    """Extracts the numeric ID portion from Waymo file naming."""
    match = re.findall(r"\d+_\d+_\d+_\d+", fname)
    return match[0] if match else None

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

def read_lidar_points(parquet_path, target_timestamp=None):
    try:
        df = pq.read_table(parquet_path).to_pandas()
    except Exception as e:
        print(f"Failed reading LiDAR parquet: {e}")
        return None

    if df.empty:
        print("LiDAR parquet empty.")
        return None

    target_timestamp = target_timestamp or df["key.frame_timestamp_micros"].iloc[0]
    df = df[df["key.frame_timestamp_micros"] == target_timestamp]

    if df.empty:
        print(f"No LiDAR for timestamp {target_timestamp}")
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

def create_box_lines(center, size, heading):
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
    return o3d.utility.Vector3dVector(corners)

def get_box_obb(center, size, heading):
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle(
        np.array([0, 0, heading])
    )
    return o3d.geometry.OrientedBoundingBox(center, R, size)

def read_lidar_boxes(parquet_path, target_timestamp=None):
    TYPE_COLOR = {1:[1,0,0], 2:[0,1,0], 3:[0,0,1], 4:[1,1,0]}
    try:
        df = pq.read_table(parquet_path).to_pandas()
    except:
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
        obbs.append(get_box_obb(c, size, heading))
        colors.append(color)
        types.append(t)

    return obbs, colors, types

def assign_box_colors(pcd, obbs, box_colors):
    points = np.asarray(pcd.points)
    final_colors = np.tile(DEFAULT_COLOR, (len(points),1))

    clustering = DBSCAN(eps=0.3, min_samples=5).fit(points)
    labels = clustering.labels_

    for obb, color in zip(obbs, box_colors):
        inside = obb.get_point_indices_within_bounding_box(pcd.points)
        if len(inside) == 0:
            continue
        touched = set(labels[inside])
        touched.discard(-1)
        for c in touched:
            final_colors[labels == c] = color
        final_colors[inside] = color

    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    return pcd

def extract_top_vehicles(pcd, obbs, box_colors, box_types, top_n=5, min_points=200):
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
        vehicles.append((veh_pcd, len(inside_idx)))

    vehicles.sort(key=lambda x: x[1], reverse=True)
    return [v[0] for v in vehicles[:top_n]]

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ALL_VEH_DIR, exist_ok=True)

    lidar_files = sorted(os.listdir(LIDAR_DIR))
    box_files   = sorted(os.listdir(BOX_DIR))

    lidar_dict = {extract_id(f): f for f in lidar_files}
    box_dict   = {extract_id(f): f for f in box_files}

    ids = sorted(set(lidar_dict.keys()) & set(box_dict.keys()))

    print(f"Found {len(ids)} matching LiDAR/box file pairs.")

    global_vehicle_counter = 1

    for id_str in ids:
        lidar_path = os.path.join(LIDAR_DIR, lidar_dict[id_str]).replace("\\", "/")
        box_path = os.path.join(BOX_DIR, box_dict[id_str]).replace("\\", "/")

        print(f"\nProcessing: {id_str}")

        pcd = read_lidar_points(lidar_path)
        print(pcd)
        obbs, colors, types = read_lidar_boxes(box_path)

        if pcd is None or len(obbs) == 0:
            print("Skipping — missing data.")
            continue

        pcd = assign_box_colors(pcd, obbs, colors)
        vehicles = extract_top_vehicles(pcd, obbs, colors, types)

        id_out_dir = os.path.join(OUTPUT_DIR, id_str)
        os.makedirs(id_out_dir, exist_ok=True)

        for i, veh in enumerate(vehicles, start=1):
            local_path = os.path.join(id_out_dir, f"vehicle_{i:02d}.ply")
            o3d.io.write_point_cloud(local_path, veh)

            global_path = os.path.join(ALL_VEH_DIR, f"vehicle_{id_str}_{global_vehicle_counter:04d}.ply")
            o3d.io.write_point_cloud(global_path, veh)
            global_vehicle_counter += 1

        print(f"Saved {len(vehicles)} vehicle clouds for ID: {id_str}")

    print("\nDONE — all vehicles extracted!")

if __name__ == "__main__":
    main()
