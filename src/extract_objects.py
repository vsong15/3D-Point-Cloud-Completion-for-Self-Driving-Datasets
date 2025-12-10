import argparse, glob, json, os, sys
from pathlib import Path
import numpy as np

# TensorFlow + Waymo
# make sure you're in (base)
conda activate base  # if not already

# upgrade pip just to be safe
python -m pip install --upgrade pip

# install CPU TensorFlow (no GPU on MacBook Air anyway)
python -m pip install "tensorflow>=2.16"
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset.utils import range_image_utils, transform_utils, frame_utils

# I/O for point clouds
import open3d as o3d


# --------------------------- helpers ---------------------------

CLASS_MAP = {
    open_dataset.Label.Type.TYPE_VEHICLE: "vehicle",
    open_dataset.Label.Type.TYPE_PEDESTRIAN: "pedestrian",
    open_dataset.Label.Type.TYPE_CYCLIST: "cyclist",
    open_dataset.Label.Type.TYPE_SIGN: "sign",
    open_dataset.Label.Type.TYPE_UNKNOWN: "unknown",
}

WANTED_TYPES = {
    "vehicle": open_dataset.Label.Type.TYPE_VEHICLE,
    "pedestrian": open_dataset.Label.Type.TYPE_PEDESTRIAN,
    "cyclist": open_dataset.Label.Type.TYPE_CYCLIST,
    "sign": open_dataset.Label.Type.TYPE_SIGN,
}

def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_ply(points_xyz: np.ndarray, out_path: str):
    """Save Nx3 float32 to .ply"""
    if points_xyz.size == 0:
        return
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
    o3d.io.write_point_cloud(out_path, pc, write_ascii=False, compressed=True)

def box_to_transform_and_extents(box):
    """
    Waymo 3D box fields:
      center_x,y,z ; length (x), width (y), height (z); heading (yaw, radians; +x forward, +y left)
    We build a homogeneous transform from box frame -> vehicle frame, and extents = (L/2, W/2, H/2).
    """
    cx, cy, cz = box.center_x, box.center_y, box.center_z
    l, w, h = box.length, box.width, box.height
    yaw = box.heading

    # Rotation about z-axis (right-handed; Waymo vehicle frame: x-forward, y-left, z-up)
    cos, sin = np.cos(yaw), np.sin(yaw)
    R = np.array([[ cos, -sin, 0.0],
                  [ sin,  cos, 0.0],
                  [0.0 ,  0.0, 1.0]], dtype=np.float32)
    t = np.array([cx, cy, cz], dtype=np.float32)

    T = np.eye(4, dtype=np.float32)
    T[:3,:3] = R
    T[:3, 3] = t
    extents = np.array([l/2.0, w/2.0, h/2.0], dtype=np.float32)
    return T, extents

def points_in_box_mask(points_xyz: np.ndarray, T_box_to_veh: np.ndarray, half_extents: np.ndarray):
    """
    Test if points (veh frame) lie inside oriented box.
    Transform points to box frame: p_box = R^T (p - t)
    Then check |x|<=lx, |y|<=ly, |z|<=lz
    """
    if points_xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)
    R = T_box_to_veh[:3,:3]
    t = T_box_to_veh[:3, 3]
    # inverse: p_box = R^T * (p - t)
    p_rel = (points_xyz - t[None, :])
    p_box = p_rel @ R  # because R is rotation; R^T == R^{-1}; here we used p @ R (equiv to R^T * p_rel^T)^T
    inside = np.all(np.abs(p_box) <= half_extents[None, :], axis=1)
    return inside

def top_lidar_point_cloud(frame: open_dataset.Frame):
    """
    Build a (N,3) XYZ point cloud from TOP LiDAR of the frame.
    Uses Waymo helpers to convert range images -> cartesian, then pick TOP sensor.
    """
    # Parse range images, camera projections, top laser intensity etc.
    (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)

    # Convert range images to point clouds for *all* lidars
    points, _ = frame_utils.convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose,
        keep_polar_features=False
    )
    # points is a list per lidar; we want TOP only
    # lidar indices in Waymo: 1=TOP, 2=FRONT, 3=SIDE_LEFT, 4=SIDE_RIGHT, 5=REAR (check calib)
    # We'll locate TOP via calibration name for robustness.
    lidar_names = {c.name: i for i, c in enumerate(frame.context.laser_calibrations)}
    # Sometimes the ordering isn't 0..4; instead use frame.lasers which are aligned with range_images keys.
    # Easiest robust way: pick the calibration with name == TOP.
    top_idx = None
    for li, calib in enumerate(frame.context.laser_calibrations):
        if calib.name == open_dataset.LaserName.TOP:
            top_idx = li
            break
    if top_idx is None:
        # Fallback: assume first
        top_idx = 0

    # points is list aligned with frame.lasers (same order as calibrations)
    pts = points[top_idx]  # (N,3)
    return np.asarray(pts, dtype=np.float32)


# --------------------------- main routine ---------------------------

def extract_objects_from_file(tfrecord_path: str, out_dir: str, keep_types: set, max_frames: int = None, start_frame_idx: int = 0):
    count_objects = 0
    frame_idx = start_frame_idx

    # Prepare class subfolders
    for cname in ["vehicle", "pedestrian", "cyclist", "sign", "unknown"]:
        ensure_dir(os.path.join(out_dir, cname))

    for rec_idx, data in enumerate(tf.data.TFRecordDataset(tfrecord_path, compression_type='')):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        # Build TOP LiDAR point cloud in vehicle frame
        try:
            xyz = top_lidar_point_cloud(frame)  # (N,3)
        except Exception as e:
            print(f"[WARN] Failed to build point cloud for frame {rec_idx}: {e}")
            continue

        # Iterate laser (LiDAR) labels (3D boxes are here)
        for lid, label in enumerate(frame.laser_labels):
            ctype = CLASS_MAP.get(label.type, "unknown")
            if keep_types and (ctype not in keep_types):
                continue

            # Get transform + extents
            T_box, half_ext = box_to_transform_and_extents(label.box)

            # Crop points
            mask = points_in_box_mask(xyz, T_box, half_ext)
            crop = xyz[mask]
            if crop.shape[0] == 0:
                continue

            # Save
            obj_id = f"f{frame_idx:06d}_obj{count_objects:05d}"
            ply_path = os.path.join(out_dir, ctype, f"{obj_id}.ply")
            meta_path = os.path.join(out_dir, ctype, f"{obj_id}.json")

            save_ply(crop, ply_path)

            meta = {
                "tfrecord": os.path.basename(tfrecord_path),
                "frame_idx": int(frame_idx),
                "object_index": int(count_objects),
                "class": ctype,
                "box": {
                    "center": [float(label.box.center_x), float(label.box.center_y), float(label.box.center_z)],
                    "size_lwh": [float(label.box.length), float(label.box.width), float(label.box.height)],
                    "heading": float(label.box.heading),
                },
                "num_points": int(crop.shape[0]),
                "ply_path": ply_path,
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

            count_objects += 1

        frame_idx += 1
        if max_frames is not None and (frame_idx - start_frame_idx) >= max_frames:
            break

    return count_objects, frame_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfrecord_glob", required=True, help='Glob for Waymo TFRecords, e.g., "/data/waymo/training/*.tfrecord"')
    ap.add_argument("--out_dir", default="objects_raw", help="Output folder for per-object crops (.ply + .json)")
    ap.add_argument("--classes", default="vehicle,pedestrian,cyclist", help="Comma list: vehicle,pedestrian,cyclist,sign,unknown")
    ap.add_argument("--max_frames", type=int, default=50, help="Max frames per TFRecord to process (to keep it quick)")
    args = ap.parse_args()

    files = sorted(glob.glob(args.tfrecord_glob))
    if not files:
        print(f"[ERR] No TFRecords matched: {args.tfrecord_glob}")
        sys.exit(1)

    keep = set()
    for c in args.classes.split(","):
        c = c.strip().lower()
        if c in WANTED_TYPES:
            keep.add(c)
        elif c == "unknown":
            keep.add(c)
        else:
            print(f"[WARN] Unknown class '{c}' ignored.")

    ensure_dir(args.out_dir)
    for c in keep:
        ensure_dir(os.path.join(args.out_dir, c))

    total = 0
    frame_cursor = 0
    for fp in files:
        print(f"[INFO] Processing {fp}")
        n, frame_cursor = extract_objects_from_file(
            tfrecord_path=fp,
            out_dir=args.out_dir,
            keep_types=keep,
            max_frames=args.max_frames,
            start_frame_idx=frame_cursor
        )
        total += n
    print(f"[DONE] Wrote {total} object crops to: {args.out_dir}")

if __name__ == "__main__":
    main()
