import numpy as np
import open3d as o3d
import pyarrow.parquet as pq
import matplotlib
import matplotlib.cm as cm

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
        
        # Color by height
        z = points[:, 2]
        min_z = np.min(z)
        max_z = np.max(z)
        norm_z = (z - min_z) / (max_z - min_z + 1e-10)
        
        cmap = matplotlib.colormaps["jet"]
        colors = cmap(norm_z)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)

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
        box_params = []
        for _, row in df.iterrows():
            obj_type = row["[LiDARBoxComponent].type"]
            color = TYPE_COLOR.get(obj_type, [1,1,1])

            center_x = -row["[LiDARBoxComponent].box.center.x"]
            center_z = row["[LiDARBoxComponent].box.center.z"] - LIDAR_HEIGHT

            SHIFT_X = 1.6
            SHIFT_Z = -3.25
            center = [
                center_x + SHIFT_X,
                row["[LiDARBoxComponent].box.center.y"],
                center_z + SHIFT_Z  
            ]

            heading = np.pi - row["[LiDARBoxComponent].box.heading"]

            size = [
                row["[LiDARBoxComponent].box.size.x"],
                row["[LiDARBoxComponent].box.size.y"],
                row["[LiDARBoxComponent].box.size.z"]
            ]

            line_set, _ = create_box_lines(center, size, heading, color)
            boxes.append(line_set)
            
            # Store params for analysis
            box_params.append({
                'center': np.array(center),
                'size': np.array(size),
                'heading': heading,
                'type': obj_type,
                'color': color
            })

        print(f"Total boxes loaded: {len(boxes)}")
        return boxes, box_params
    except Exception as e:
        print(f"Error reading LiDAR boxes: {e}")
        return [], []

def get_local_points(points, center, heading):
    diff = points - center
    c = np.cos(heading)
    s = np.sin(heading)
    # Inverse rotation (transpose)
    x = diff[:, 0] * c + diff[:, 1] * s
    y = diff[:, 0] * -s + diff[:, 1] * c
    z = diff[:, 2]
    return np.stack([x, y, z], axis=-1)

def align_boxes(pcd, box_params):
    if pcd is None or not box_params:
        return []

    points = np.asarray(pcd.points)
    aligned_geometries = []
    
    print("\n--- Auto-Aligning Boxes ---")
    
    for i, box in enumerate(box_params):
        center = box['center']
        size = box['size']
        heading = box['heading']
        color = box.get('color', [1, 0, 0])
        obj_type = box['type']
        
        # Only align vehicles (Type 1)
        if obj_type == 1:
            # --- Z Alignment ---
            # Create a tall search region (column) to find points above/below
            R = o3d.geometry.OrientedBoundingBox.get_rotation_matrix_from_axis_angle(np.array([0, 0, heading]))
            
            # Search column: slightly larger X/Y to catch points just outside, tall Z
            search_size = size.copy()
            search_size[0] += 1.0 # Expand search in X
            search_size[1] += 1.0 # Expand search in Y
            search_size[2] = 5.0 
            
            # Center the search column at the current box center
            obb = o3d.geometry.OrientedBoundingBox(center, R, search_size)
            
            # Get points inside this column
            indices = obb.get_point_indices_within_bounding_box(pcd.points)
            
            if len(indices) >= 10:
                box_points = points[indices]
                
                # 1. Z Alignment
                min_z_points = np.percentile(box_points[:, 2], 1)
                current_box_bottom = center[2] - size[2]/2
                diff_z = min_z_points - current_box_bottom
                
                if abs(diff_z) < 2.0: 
                    center[2] += diff_z
                    print(f"Vehicle {i}: Adjusted Z by {diff_z:.3f}m")
                
                # 2. X/Y Alignment (Containment)
                # Transform points to local box coordinates
                local_pts = get_local_points(box_points, center, heading)
                
                # Box boundaries in local coords
                min_bx, max_bx = -size[0]/2, size[0]/2
                min_by, max_by = -size[1]/2, size[1]/2
                
                # Point boundaries (robust)
                min_px = np.percentile(local_pts[:, 0], 1)
                max_px = np.percentile(local_pts[:, 0], 99)
                min_py = np.percentile(local_pts[:, 1], 1)
                max_py = np.percentile(local_pts[:, 1], 99)
                
                # Check for overflow (points outside box)
                # If points are significantly outside, shift box to cover them
                shift_x = 0
                shift_y = 0
                
                # X Axis
                if min_px < min_bx: # Overflow Left
                    shift_x += (min_px - min_bx)
                if max_px > max_bx: # Overflow Right
                    shift_x += (max_px - max_bx)
                    
                # Y Axis
                if min_py < min_by: # Overflow Back
                    shift_y += (min_py - min_by)
                if max_py > max_by: # Overflow Front
                    shift_y += (max_py - max_by)
                
                # Apply shifts if they are reasonable
                if abs(shift_x) > 0.05 and abs(shift_x) < 1.0:
                    # Rotate shift back to global
                    global_shift_x = shift_x * np.cos(heading) - shift_y * np.sin(heading)
                    global_shift_y = shift_x * np.sin(heading) + shift_y * np.cos(heading)
                    
                    center[0] += global_shift_x
                    center[1] += global_shift_y
                    print(f"Vehicle {i}: Adjusted X/Y by local ({shift_x:.3f}, {shift_y:.3f})")

        # Recreate geometry with (potentially) updated center
        line_set, _ = create_box_lines(center, size, heading, color)
        aligned_geometries.append(line_set)

    return aligned_geometries

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
    _, box_params = read_lidar_boxes(boxes_path, timestamp)
    
    aligned_boxes = align_boxes(pcd, box_params)
    
    visualize(pcd, aligned_boxes)

if __name__ == "__main__":
    main()
    