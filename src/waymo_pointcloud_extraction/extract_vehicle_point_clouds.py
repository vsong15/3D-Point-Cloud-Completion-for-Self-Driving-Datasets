import os
import re
import numpy as np
import open3d as o3d
import pyarrow.parquet as pq
from sklearn.cluster import DBSCAN
import matplotlib
import matplotlib.cm as cm

LIDAR_DIR = "data/training/254_training_lidar"
BOX_DIR = "data/training/254_training_lidar_box"
OUTPUT_DIR = "vehicles_out/vehicles_out_50_min_points_updated_bb"
ALL_VEH_DIR = "waymo_vehicle_incomplete_pointclouds/all_vehicle_point_clouds_50_min_points_updated_bb"

LIDAR_HEIGHT = 2.0
SHIFT_X = 1.6
SHIFT_Z = -3.25
DEFAULT_COLOR = np.array([0.6, 0.6, 0.6])
MIN_POINTS = 50
TOP_N = 5

def extract_id(fname):
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
    except:
        return None

def read_lidar_points(parquet_path, target_timestamp=None):
    try:
        df = pq.read_table(parquet_path).to_pandas()
        if df.empty:
            print(f"LiDAR parquet empty: {parquet_path}")
            return None
    except:
        print(f"Failed to read LiDAR parquet: {parquet_path}")
        return None
    target_timestamp = target_timestamp or df["key.frame_timestamp_micros"].iloc[0]
    df = df[df["key.frame_timestamp_micros"] == target_timestamp]
    if df.empty:
        print(f"No LiDAR data for timestamp {target_timestamp}")
        return None
    all_pts = []
    for idx in range(len(df)):
        try:
            x = df["[LiDARComponent].cartesian.x"].iloc[idx]
            y = df["[LiDARComponent].cartesian.y"].iloc[idx]
            z = df["[LiDARComponent].cartesian.z"].iloc[idx]
            all_pts.append(np.stack([x, y, z], axis=-1))
            continue
        except: pass
        for rname in ["range_image_return1", "range_image_return2"]:
            vals = df[f"[LiDARComponent].{rname}.values"].iloc[idx]
            shape = df[f"[LiDARComponent].{rname}.shape"].iloc[idx]
            if vals is not None and shape is not None:
                xyz = process_range_image(vals, shape)
                if xyz is not None: all_pts.append(xyz)
    if not all_pts:
        print("No LiDAR points found.")
        return None
    points = np.vstack(all_pts)
    points[:, 2] -= LIDAR_HEIGHT
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd.paint_uniform_color(DEFAULT_COLOR)
    pcd = pcd.voxel_down_sample(0.05)
    print(f"Total LiDAR points loaded: {len(pcd.points)}")
    return pcd

def get_box_obb(center, size, heading):
    R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle(np.array([0, 0, heading]))
    return o3d.geometry.OrientedBoundingBox(center, R, size)

def read_lidar_boxes(parquet_path, target_timestamp=None):
    TYPE_COLOR = {1:[1,0,0], 2:[0,1,0], 3:[0,0,1], 4:[1,1,0]}
    try:
        df = pq.read_table(parquet_path).to_pandas()
        if df.empty:
            print(f"No boxes found: {parquet_path}")
            return [], [], []
    except:
        print(f"Failed to read box parquet: {parquet_path}")
        return [], [], []
    target_timestamp = target_timestamp or df["key.frame_timestamp_micros"].iloc[0]
    df = df[df["key.frame_timestamp_micros"] == target_timestamp]
    obbs, colors, types = [], [], []
    for _, row in df.iterrows():
        center = np.array([-row["[LiDARBoxComponent].box.center.x"] + SHIFT_X,
                            row["[LiDARBoxComponent].box.center.y"],
                            row["[LiDARBoxComponent].box.center.z"] - LIDAR_HEIGHT + SHIFT_Z])
        size = np.array([row["[LiDARBoxComponent].box.size.x"],
                         row["[LiDARBoxComponent].box.size.y"],
                         row["[LiDARBoxComponent].box.size.z"]])
        heading = np.pi - row["[LiDARBoxComponent].box.heading"]
        t = row["[LiDARBoxComponent].type"]
        obbs.append(get_box_obb(center, size, heading))
        colors.append(TYPE_COLOR.get(t, [1,1,1]))
        types.append(t)
    print(f"Total boxes loaded: {len(obbs)}")
    return obbs, colors, types

def assign_box_colors(pcd, obbs, box_colors):
    points = np.asarray(pcd.points)
    final_colors = np.tile(DEFAULT_COLOR, (len(points),1))
    clustering = DBSCAN(eps=0.3, min_samples=5).fit(points)
    labels = clustering.labels_
    for obb, color in zip(obbs, box_colors):
        inside = obb.get_point_indices_within_bounding_box(pcd.points)
        if not inside: continue
        touched = set(labels[inside])
        touched.discard(-1)
        for c in touched:
            final_colors[labels == c] = color
        final_colors[inside] = color
    pcd.colors = o3d.utility.Vector3dVector(final_colors)
    return pcd

def extract_top_vehicles(pcd, obbs, box_colors, box_types, top_n=TOP_N, min_points=MIN_POINTS):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    vehicles = []
    for obb, color, t in zip(obbs, box_colors, box_types):
        if t != 1: continue
        inside_idx = obb.get_point_indices_within_bounding_box(pcd.points)
        if len(inside_idx) < min_points: continue
        veh_pcd = o3d.geometry.PointCloud()
        veh_pcd.points = o3d.utility.Vector3dVector(points[inside_idx])
        veh_pcd.colors = o3d.utility.Vector3dVector(colors[inside_idx])
        vehicles.append((veh_pcd, len(inside_idx)))
    vehicles.sort(key=lambda x: x[1], reverse=True)
    print(f"Top {len(vehicles[:top_n])} vehicles extracted")
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
        print(f"\nProcessing ID: {id_str}")
        lidar_path = os.path.join(LIDAR_DIR, lidar_dict[id_str]).replace("\\","/")
        box_path = os.path.join(BOX_DIR, box_dict[id_str]).replace("\\","/")
        pcd = read_lidar_points(lidar_path)
        obbs, colors, types = read_lidar_boxes(box_path)
        if pcd is None or not obbs:
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
            print(f"Saved vehicle {i} for ID {id_str} — global count: {global_vehicle_counter}")
            global_vehicle_counter += 1
    print("\nDONE — all vehicles extracted!")

if __name__ == "__main__":
    main()
