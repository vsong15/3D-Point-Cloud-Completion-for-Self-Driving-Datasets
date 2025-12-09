import numpy as np
import open3d as o3d
import pyarrow.parquet as pq
import os

def create_bounding_box(center, dimensions, heading, color=[1, 1, 1]):
    """
    Create an Open3D bounding box with enhanced visibility
    Args:
        center: [x, y, z] center coordinates
        dimensions: [length, width, height]
        heading: rotation around Z-axis in radians
        color: RGB color for the box lines
    """
    # Create rotation matrix from heading (rotate around Z-axis)
    R = np.array([
        [np.cos(heading), -np.sin(heading), 0],
        [np.sin(heading), np.cos(heading), 0],
        [0, 0, 1]
    ])
    
    # Create oriented bounding box
    box = o3d.geometry.OrientedBoundingBox(
        center=center,
        R=R,
        extent=dimensions
    )
    
    # Create points for the box corners
    points = box.get_box_points()
    
    # Define lines for edges and diagonals
    lines = [
        # Bottom face
        [0, 1], [1, 2], [2, 3], [3, 0],
        # Top face
        [4, 5], [5, 6], [6, 7], [7, 4],
        # Vertical edges
        [0, 4], [1, 5], [2, 6], [3, 7],
        # Diagonal lines to show orientation (front face)
        [0, 2], [1, 3],  # Bottom face diagonals
        [4, 6], [5, 7],  # Top face diagonals
    ]
    
    # Create line colors (brighter for edges, slightly darker for diagonals)
    colors = []
    for i in range(len(lines)):
        if i < 12:  # Main edges
            colors.append(color)
        else:  # Diagonals
            # Make diagonals slightly darker
            darker_color = [c * 0.7 for c in color]
            colors.append(darker_color)
    
    # Create line set with the specified colors
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points)),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def create_coordinate_frame(size=10):
    """Create a coordinate frame"""
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)

def create_ground_grid(size=100, step=10):
    """Create a ground plane grid"""
    # Create lines for the grid
    points = []
    lines = []
    colors = []
    
    # Create grid lines
    for i in range(-size, size+1, step):
        points.append([i, -size, 0])
        points.append([i, size, 0])
        points.append([-size, i, 0])
        points.append([size, i, 0])
        
        lines.append([len(points)-4, len(points)-3])
        lines.append([len(points)-2, len(points)-1])
        
        colors.extend([[0.5, 0.5, 0.5] for _ in range(2)])  # Grey color for grid
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set

def visualize_point_cloud_with_boxes(points, boxes_df, boxes_parquet_path, timestamp=None, mode='all', color_mode='height'):
    """
    Visualize point cloud with 3D bounding boxes from parquet data
    Args:
        points: Nx3 numpy array of point cloud points
        boxes_df: DataFrame containing box data
        boxes_parquet_path: Path to original parquet file (for schema inspection)
        timestamp: Optional timestamp to filter boxes
        mode: 'all' or 'inside' visualization mode
        color_mode: 'height' or 'box_type' coloring mode
    """
    # Convert points to numpy array if not already
    points_np = np.asarray(points)
    
    # Verify data structure integrity
    required_cols = [
        '[LiDARBoxComponent].box.center.x',
        '[LiDARBoxComponent].box.center.y',
        '[LiDARBoxComponent].box.center.z',
        '[LiDARBoxComponent].box.size.x',
        '[LiDARBoxComponent].box.size.y',
        '[LiDARBoxComponent].box.size.z',
        '[LiDARBoxComponent].box.heading',
        '[LiDARBoxComponent].type',
        '[LiDARBoxComponent].num_lidar_points_in_box'
    ]
    
    missing_cols = [col for col in required_cols if col not in boxes_df.columns]
    if missing_cols:
        print("ERROR: Missing required columns in box data:")
        for col in missing_cols:
            print(f"- {col}")
        return
        
    # Verify point cloud structure
    if not isinstance(points, np.ndarray) or points.shape[1] != 3:
        print(f"ERROR: Invalid point cloud structure. Expected Nx3 array, got shape: {points.shape if isinstance(points, np.ndarray) else type(points)}")
        return
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    
    # Color points by height with a better color scheme
    min_z = np.min(points_np[:, 2])
    max_z = np.max(points_np[:, 2])
    
    # Normalize height to [0, 1] but clip very low points
    normalized_z = (points[:, 2] - min_z) / (max_z - min_z + 1e-6)
    ground_threshold = 0.1  # Adjust this value to better separate ground from objects
    
    colors = np.zeros((len(points), 3))
    
    # Ground points (yellow-orange)
    ground_mask = normalized_z < ground_threshold
    colors[ground_mask] = [1.0, 0.7, 0.0]  # Orange for ground
    
    # Above-ground points (gradient from cyan to blue)
    above_ground = ~ground_mask
    colors[above_ground, 0] = 0.0  # No red for above-ground
    colors[above_ground, 1] = 1 - (normalized_z[above_ground] - ground_threshold) / (1 - ground_threshold)  # Green decreases with height
    colors[above_ground, 2] = 1.0  # Full blue for above-ground
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Print raw parquet schema and first few rows
    print("\nParquet file schema:")
    table = pq.read_table(boxes_parquet_path)
    print(table.schema)
    
    print("\nSample of raw parquet data (first row):")
    df_raw = table.to_pandas()
    print(df_raw.iloc[0].to_string())
    
    # Filter boxes for the specific timestamp if provided
    if timestamp is not None:
        boxes_df = boxes_df[boxes_df['key.frame_timestamp_micros'] == timestamp]
        
    # Print column names we're using for box data
    print("\nBox-related columns found:")
    box_cols = [col for col in boxes_df.columns if any(term in col for term in ['center', 'size', 'heading', 'type', 'num_lidar'])]
    for col in box_cols:
        print(f"- {col}")
    
    # Filter out boxes with very few points
    boxes_df = boxes_df[boxes_df['[LiDARBoxComponent].num_lidar_points_in_box'] > 10]
    
    print(f"\nProcessing {len(boxes_df)} boxes with >10 points for timestamp {timestamp}")
    
    # Compare coordinate ranges between points and boxes
    point_ranges = {
        'x': (points_np[:, 0].min(), points_np[:, 0].max(), points_np[:, 0].mean()),
        'y': (points_np[:, 1].min(), points_np[:, 1].max(), points_np[:, 1].mean()),
        'z': (points_np[:, 2].min(), points_np[:, 2].max(), points_np[:, 2].mean())
    }
    
    box_ranges = {
        'x': (boxes_df['[LiDARBoxComponent].box.center.x'].min(), 
              boxes_df['[LiDARBoxComponent].box.center.x'].max(),
              boxes_df['[LiDARBoxComponent].box.center.x'].mean()),
        'y': (boxes_df['[LiDARBoxComponent].box.center.y'].min(),
              boxes_df['[LiDARBoxComponent].box.center.y'].max(),
              boxes_df['[LiDARBoxComponent].box.center.y'].mean()),
        'z': (boxes_df['[LiDARBoxComponent].box.center.z'].min(),
              boxes_df['[LiDARBoxComponent].box.center.z'].max(),
              boxes_df['[LiDARBoxComponent].box.center.z'].mean())
    }
    
    print("\nCoordinate system comparison:")
    for axis in ['x', 'y', 'z']:
        print(f"\n{axis.upper()} axis:")
        print(f"  Points  - min: {point_ranges[axis][0]:.2f}, max: {point_ranges[axis][1]:.2f}, mean: {point_ranges[axis][2]:.2f}")
        print(f"  Boxes   - min: {box_ranges[axis][0]:.2f}, max: {box_ranges[axis][1]:.2f}, mean: {box_ranges[axis][2]:.2f}")
        print(f"  Offset  - mean box pos - mean point pos: {box_ranges[axis][2] - point_ranges[axis][2]:.2f}")
    
    print("\nBox types distribution:")
    type_counts = boxes_df['[LiDARBoxComponent].type'].value_counts()
    for type_id, count in type_counts.items():
        type_name = {1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4: "Cyclist"}.get(type_id, "Unknown")
        print(f"- {type_name}: {count} boxes")
    
    # Create visualization geometries
    geometries = []
    
    # Add coordinate frame
    coord_frame = create_coordinate_frame(size=5)
    geometries.append(coord_frame)
    
    # Add ground grid
    ground_grid = create_ground_grid(size=50, step=5)
    geometries.append(ground_grid)
    
    # Prepare point cloud (we may filter points later based on mode)
    points_np = np.asarray(points)
    print(f"Point cloud stats: min={points_np.min(axis=0)}, max={points_np.max(axis=0)}, mean={points_np.mean(axis=0)}")
    # Print box stats
    box_centers = np.array([
        boxes_df['[LiDARBoxComponent].box.center.x'],
        boxes_df['[LiDARBoxComponent].box.center.y'],
        boxes_df['[LiDARBoxComponent].box.center.z']
    ]).T
    box_dims = np.array([
        boxes_df['[LiDARBoxComponent].box.size.x'],
        boxes_df['[LiDARBoxComponent].box.size.y'],
        boxes_df['[LiDARBoxComponent].box.size.z']
    ]).T
    print(f"Box center stats: min={box_centers.min(axis=0)}, max={box_centers.max(axis=0)}, mean={box_centers.mean(axis=0)}")
    print(f"Box dims stats: min={box_dims.min(axis=0)}, max={box_dims.max(axis=0)}, mean={box_dims.mean(axis=0)}")
    print(f"Mean box center: {box_centers.mean(axis=0)}")
    print(f"Mean point cloud center: {points_np.mean(axis=0)}")
    print(f"Offset (box center - point cloud center): {box_centers.mean(axis=0) - points_np.mean(axis=0)}")
    geometries.append(pcd)
    
    # Print available columns for debugging
    print("\nAvailable columns in boxes_df:")
    print(boxes_df.columns.tolist())
    
    # First, let's check raw distances between points and box centers
    distances_to_boxes = []
    for center in box_centers:
        # Calculate Euclidean distance from each point to box center
        dist = np.linalg.norm(points_np - center, axis=1)
        min_dist = np.min(dist)
        distances_to_boxes.append(min_dist)
    
    distances_to_boxes = np.array(distances_to_boxes)
    print(f"\nBox-to-nearest-point distances (meters):")
    print(f"Min: {np.min(distances_to_boxes):.2f}")
    print(f"Max: {np.max(distances_to_boxes):.2f}")
    print(f"Mean: {np.mean(distances_to_boxes):.2f}")
    print(f"Median: {np.median(distances_to_boxes):.2f}")
    print(f"Number of boxes: {len(distances_to_boxes)}")
    print(f"Boxes with points within 5m: {np.sum(distances_to_boxes < 5.0)}/{len(distances_to_boxes)}")
    
    # Prepare mask for points inside any box (used when mode == 'inside')
    points_in_any_box = np.zeros(len(points_np), dtype=bool)
    # Prepare per-point box-type assignment (-1 = none)
    points_box_type = np.full(len(points_np), -1, dtype=int)

    # No offset applied; use raw coordinates for diagnostics
    box_centers = np.array([
        boxes_df['[LiDARBoxComponent].box.center.x'],
        boxes_df['[LiDARBoxComponent].box.center.y'],
        boxes_df['[LiDARBoxComponent].box.center.z']
    ]).T
    print("Box centers shape:", box_centers.shape)
    print("Point cloud shape:", points.shape)
    # Print XY range for both
    print(f"Box centers XY min: {box_centers[:, :2].min(axis=0)}, max: {box_centers[:, :2].max(axis=0)}")
    print(f"Point cloud XY min: {points[:, :2].min(axis=0)}, max: {points[:, :2].max(axis=0)}")
    # Plot XY scatter for both
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], s=1, alpha=0.2, label='Point Cloud')
    plt.scatter(box_centers[:, 0], box_centers[:, 1], s=40, c='red', marker='x', label='Box Centers')
    plt.legend()
    plt.title('XY Distribution: Point Cloud vs Box Centers')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()
    # Padding (in meters) to inflate boxes for point-in-box testing to account for LiDAR sparsity
    PAD = 0.5
    print(f"Using padding={PAD} m for box inflation when testing points-inside-boxes.")
    # Build a KD-tree for nearest-neighbor lookup from point cloud to box centers
    try:
        pcd_tree = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_np)))
        print("KD-tree built for point cloud. Nearest neighbor distances to box centers:")
        for i, c in enumerate(box_centers[:10]):
            [k, idx, dist2] = pcd_tree.search_knn_vector_3d(c, 1)
            if k > 0:
                d = np.sqrt(dist2[0])
            else:
                d = float('nan')
            print(f"  Box {i}: center={c}, nearest_point_dist={d:.3f} m, nearest_idx={idx[0] if k>0 else None}")
    except Exception as e:
        print(f"KD-tree nearest neighbor check failed: {e}")
    # Process each box
    for i, (idx, box_row) in enumerate(boxes_df.iterrows()):
            # Extract box parameters using correct column names
            center = np.array([
                box_row['[LiDARBoxComponent].box.center.x'],
                box_row['[LiDARBoxComponent].box.center.y'],
                box_row['[LiDARBoxComponent].box.center.z']
            ])
            dimensions = np.array([
                box_row['[LiDARBoxComponent].box.size.x'],
                box_row['[LiDARBoxComponent].box.size.y'],
                box_row['[LiDARBoxComponent].box.size.z']
            ])
            heading = box_row['[LiDARBoxComponent].box.heading']

            # Print debug info for first 3 boxes
            if i < 3:
                print(f"\n--- Box {i} diagnostics ---")
                print(f"Box {i}: center={center}, dimensions={dimensions}, heading={heading}")
                z_extent = dimensions[2]
                z_mask = np.abs(points_np[:, 2] - center[2]) <= (z_extent * 2)
                near_z_points = points_np[z_mask]
                print(f"  Z stats of points within 2x box Z extent:")
                if len(near_z_points) > 0:
                    print(f"    min={near_z_points[:,2].min()}, max={near_z_points[:,2].max()}, mean={near_z_points[:,2].mean()}, count={len(near_z_points)}")
                else:
                    print("    No points within 2x box Z extent.")

            # Determine color based on object type
            # Waymo types: 1 = Vehicle, 2 = Pedestrian, 3 = Sign, 4 = Cyclist
            box_type = box_row['[LiDARBoxComponent].type']
            if box_type == 1:
                color = [1, 0.7, 0]  # Orange for vehicles
            elif box_type == 2:
                color = [0, 1, 0]  # Green for pedestrians
            elif box_type == 4:
                color = [0, 1, 1]  # Cyan for cyclists
            else:
                color = [1, 1, 0]  # Yellow for others

            # Create Open3D box line set with color
            o3d_box_lines = create_bounding_box(center, dimensions, heading, color)
            geometries.append(o3d_box_lines)

            # Compute which points are inside this box (analytical test)
            # Rotation matrix R (same as used to create the box)
            R = np.array([
                [np.cos(heading), -np.sin(heading), 0],
                [np.sin(heading),  np.cos(heading), 0],
                [0, 0, 1]
            ])
            # Transform points to box-local coordinates
            # If world = R @ local + center, then local = R.T @ (world - center).
            # For row-vector batches this is equivalent to (world - center) @ R.
            # Use R (not R.T) on the right to compute local coordinates for each row.
            local = (points_np - center) @ R
            # Be more lenient with point-in-box test since LiDAR often misses edges
            # Use 1.5x the box dimensions laterally (XY) and 2x vertically (Z)
            half = dimensions / 2.0
            scaled_half = np.array([half[0] * 1.5, half[1] * 1.5, half[2] * 2.0])
            inside_mask = np.all(np.abs(local) <= scaled_half, axis=1)
            if i < 3:
                print(f"Box {i}: points inside = {inside_mask.sum()} (original box)")
                # Show local coords for the nearest 5 points to the box center (more informative than first 5)
                try:
                    k, idxs, d2 = pcd_tree.search_knn_vector_3d(center, 5)
                    nearest_idxs = idxs[:k]
                    print(f"  Nearest {k} point indices: {nearest_idxs}")
                    print(f"  Nearest points (world coords): {points_np[nearest_idxs]}")
                    print(f"  Local coords of nearest points: {local[nearest_idxs]}")
                except Exception:
                    # Fallback to first 5 if KD-tree not available
                    print(f"  Sample local coords (first 5): {local[:5]}")
                # Diagnostic: count points within 2x, 3x, 5x, 10x half-extent
                for mult in [2, 3, 5, 10]:
                    mask = np.all(np.abs(local) <= (half * mult), axis=1)
                    print(f"  Points within {mult}x half-extent: {mask.sum()}")
                # Test with 2x and 3x box extents for point-in-box
                inside_mask_2x = np.all(np.abs(local) <= (half * 2), axis=1)
                inside_mask_3x = np.all(np.abs(local) <= (half * 3), axis=1)
                print(f"  Points inside 2x box extents: {inside_mask_2x.sum()}")
                print(f"  Points inside 3x box extents: {inside_mask_3x.sum()}")
                # Now test with an explicit PAD (inflated box)
                half_padded = (dimensions / 2.0) + PAD
                inside_padded = np.all(np.abs(local) <= half_padded, axis=1)
                print(f"  Points inside with PAD={PAD} m: {inside_padded.sum()}")
                # Print mean/std of local coordinates for points near box in Z
                z_extent = dimensions[2]
                z_mask = np.abs(points_np[:, 2] - center[2]) <= (z_extent * 2)
                local_near_z = local[z_mask]
                if len(local_near_z) > 0:
                    print(f"  Local coords near box Z: mean={local_near_z.mean(axis=0)}, std={local_near_z.std(axis=0)}")
                else:
                    print("  No local coords near box Z.")
            points_in_any_box |= inside_mask
            # assign box type to points not yet assigned
            unassigned = (points_box_type == -1)
            assign_mask = inside_mask & unassigned
            points_box_type[assign_mask] = box_type
            t = points_box_type[i]
            if t != -1:
                colors_np[i] = type_to_color.get(int(t), [1, 1, 0])
    # Only set colors_np if defined (box_type mode)
    if color_mode == 'box_type':
        pcd.colors = o3d.utility.Vector3dVector(colors_np)

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # If mode == 'inside', replace point cloud with only points inside boxes
    if mode == 'inside':
        inside_count = int(points_in_any_box.sum())
        print(f"Points inside boxes: {inside_count}")
        if inside_count == 0:
            print("No points found inside boxes; visualizing boxes only.")
        else:
            # Update point cloud to only contain inside points
            inside_idx = points_in_any_box
            # If color_mode == 'box_type', color points by their assigned box type
            if color_mode == 'box_type':
                # Map types to colors
                type_to_color = {1: [1, 0.7, 0], 2: [0, 1, 0], 4: [0, 1, 1]}
                inside_colors = []
                for t in points_box_type[inside_idx]:
                    c = type_to_color.get(int(t), [1, 1, 0]) if t != -1 else [0.5, 0.5, 0.5]
                    inside_colors.append(c)
                pcd.points = o3d.utility.Vector3dVector(points_np[inside_idx])
                pcd.colors = o3d.utility.Vector3dVector(np.array(inside_colors))
            else:
                # Default: keep existing coloring but filter to inside points
                colors_np = np.asarray(pcd.colors)
                if colors_np.shape[0] == len(points_np):
                    pcd.points = o3d.utility.Vector3dVector(points_np[inside_idx])
                    pcd.colors = o3d.utility.Vector3dVector(colors_np[inside_idx])
                else:
                    pcd.points = o3d.utility.Vector3dVector(points_np[inside_idx])

    # Add geometries
    for geom in geometries:
        vis.add_geometry(geom)
    
    # Set up the view
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])  # Pure black background
    opt.point_size = 2.5  # Adjusted point size
    opt.line_width = 3.0  # Thicker lines for boxes
    opt.point_show_normal = False
    opt.show_coordinate_frame = True
    
    # Set initial camera view (bird's eye view slightly angled)
    ctr = vis.get_view_control()
    ctr.set_zoom(0.45)  # Slightly closer view
    ctr.set_front([-0.3, -0.3, -0.9])  # More top-down view
    ctr.set_lookat([0, 0, 0])  # Looking at the center
    ctr.set_up([0, 0, 1])  # Z-axis is up
    
    # Run the visualizer
    vis.run()
    vis.destroy_window()

def read_boxes(box_parquet_path, timestamp=None):
    """
    Read bounding box data from parquet file
    """
    try:
        print(f"\nReading boxes from: {box_parquet_path}")
        table = pq.read_table(box_parquet_path)
        boxes_df = table.to_pandas()
        
        print(f"\nParquet file schema:")
        print(table.schema)
        
        print("\nAvailable columns:")
        for col in boxes_df.columns:
            print(f"- {col}")
            
        print("\nFirst row of data:")
        print(boxes_df.iloc[0].to_dict())
        
        print(f"\nTotal boxes loaded: {len(boxes_df)}")
        if timestamp is not None:
            boxes_df = boxes_df[boxes_df['key.frame_timestamp_micros'] == timestamp]
            print(f"Boxes for timestamp {timestamp}: {len(boxes_df)}")
        
        return boxes_df
        
    except Exception as e:
        print(f"Error reading box parquet file: {e}")
        print(f"Exception details: {str(e)}")
        return None

def main():
    import sys
    if len(sys.argv) < 3:
        print("Please provide both point cloud parquet and box parquet paths")
        print("Usage: python visualize_with_boxes.py <points_parquet_path> <boxes_parquet_path> [timestamp] [mode] [color_mode]")
        print("mode: 'all' (default) show full point cloud and boxes, 'inside' show only boxes and points inside boxes")
        print("color_mode: 'height' (default) color points by height, 'box_type' color points by the type of the box they belong to")
        return
        
    points_parquet_path = sys.argv[1]
    boxes_parquet_path = sys.argv[2]
    timestamp = int(sys.argv[3]) if len(sys.argv) > 3 else None
    mode = sys.argv[4] if len(sys.argv) > 4 else 'all'
    color_mode = sys.argv[5] if len(sys.argv) > 5 else 'height'
    
    # First read points using existing dataset_read.py
    from dataset_read import read_and_process_lidar
    points = read_and_process_lidar(points_parquet_path, timestamp)
    print(f"Loaded point cloud shape: {None if points is None else points.shape}")
    if points is not None:
        print("Sample points (first 5):")
        print(points[:5])
    if points is None:
        print("Failed to load point cloud data")
        return
        
    # Read bounding boxes
    boxes_df = read_boxes(boxes_parquet_path, timestamp)
    
    if boxes_df is None or len(boxes_df) == 0:
        print("Failed to load bounding box data")
        return
        
    # Visualize points with boxes
    visualize_point_cloud_with_boxes(points, boxes_df, boxes_parquet_path, timestamp, mode=mode, color_mode=color_mode)

if __name__ == "__main__":
    main()