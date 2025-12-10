import numpy as np
import open3d as o3d
import pyarrow.parquet as pq
import os

def spherical_to_cartesian(range_image, pixel_pose):
    """
    Convert spherical coordinates to cartesian coordinates for Waymo dataset.
    The range image channels are [range, intensity, elongation, timestamp]
    """
    height, width, _ = range_image.shape
    
    # Create coordinate grid
    az_correction = np.zeros((height, width))
    ratios = (0.5 + np.arange(width)) / width
    az_correction = (ratios - 0.5) * 2 * np.pi
    
    ratios = (0.5 + np.arange(height)) / height
    incl = np.expand_dims(ratios * np.pi, 1)  # shape (64, 1)
    
    # Get range values
    ranges = range_image[..., 0]
    
    # Calculate coordinates
    x = ranges * np.cos(incl) * np.cos(az_correction)
    y = ranges * np.cos(incl) * np.sin(az_correction)
    z = ranges * np.sin(incl)
    
    # Stack coordinates
    points = np.stack([x, y, z], axis=-1)
    
    # Mask out invalid points (range = 0 or infinity)
    valid_mask = ranges > 0
    points = points[valid_mask]
    
    return points

def process_range_image(values, shape):
    """
    Process Waymo range image values into point cloud coordinates
    """
    try:
        # Convert values to numpy array and reshape
        points = np.array(values).reshape(shape[0], shape[1], shape[2])
        print(f"Range image shape: {points.shape}")
        
        # Convert spherical coordinates to cartesian
        # The range image contains [range, intensity, elongation, timestamp]
        cartesian_points = spherical_to_cartesian(points, None)  # No pose information for now
        
        if len(cartesian_points) > 0:
            print(f"Converted {len(cartesian_points)} valid points")
            return cartesian_points
        return None
        
    except Exception as e:
        print(f"Error processing range image: {e}")
        print(f"Exception details: {str(e)}")
        return None

def read_and_process_lidar(parquet_path, target_timestamp=None):
    """
    Read Waymo LiDAR data from parquet file and convert to point cloud
    Args:
        parquet_path: Path to the parquet file
        target_timestamp: Specific timestamp to process (in microseconds). 
                         If None, processes the first timestamp.
    """
    try:
        # Read the parquet file
        table = pq.read_table(parquet_path)
        df = table.to_pandas()

        # Print available columns for debugging (helpful to verify parquet layout)
        print("\nPoint parquet columns:")
        for col in df.columns:
            print(f"- {col}")

        # Get unique timestamps
        timestamps = df['key.frame_timestamp_micros'].unique()
        print(f"\nFound {len(timestamps)} timestamps in the dataset:")
        for ts in timestamps:
            print(f"Timestamp: {ts} microseconds")
        
        # If no specific timestamp is provided, use the first one
        if target_timestamp is None:
            target_timestamp = timestamps[0]
            print(f"\nUsing first timestamp: {target_timestamp}")
        elif target_timestamp not in timestamps:
            print(f"Timestamp {target_timestamp} not found in dataset")
            return None
            
        # Filter data for the target timestamp
        df_timestamp = df[df['key.frame_timestamp_micros'] == target_timestamp]
        print(f"\nProcessing data for timestamp: {target_timestamp}")
        print(f"Number of laser returns: {len(df_timestamp)}")
        
        all_points = []
        
        # Process each laser for this timestamp
        for idx in range(len(df_timestamp)):
            laser_name = df_timestamp['key.laser_name'].iloc[idx]
            print(f"\nProcessing laser {laser_name}...")
            # Print raw range-image shape info for diagnostics
            try:
                v1 = df_timestamp['[LiDARComponent].range_image_return1.values'].iloc[idx]
                s1 = df_timestamp['[LiDARComponent].range_image_return1.shape'].iloc[idx]
                print(f"  return1 shape: {s1}, values length: {None if v1 is None else len(v1)}")
            except Exception:
                pass

            # Process first returns
            values1 = df_timestamp['[LiDARComponent].range_image_return1.values'].iloc[idx]
            shape1 = df_timestamp['[LiDARComponent].range_image_return1.shape'].iloc[idx]
            print(f"Processing return1 range image...")
            points1 = process_range_image(values1, shape1)
            if points1 is not None:
                all_points.append(points1)

            # Process second returns if they exist
            values2 = df_timestamp['[LiDARComponent].range_image_return2.values'].iloc[idx]
            if len(values2) > 0:  # Only process if there are second returns
                shape2 = df_timestamp['[LiDARComponent].range_image_return2.shape'].iloc[idx]
                print(f"Processing return2 range image...")
                points2 = process_range_image(values2, shape2)
                if points2 is not None:
                    all_points.append(points2)
        
        # Combine all points
        if all_points:
            combined_points = np.vstack(all_points)
            print(f"\nTotal points for timestamp {target_timestamp}: {len(combined_points)}")
            return combined_points
        # Fallback: if dataframe contains explicit XYZ columns, use them
        # (some Waymo/parquet exports include flattened point lists)
        if 'x' in df_timestamp.columns and 'y' in df_timestamp.columns and 'z' in df_timestamp.columns:
            try:
                pts = df_timestamp[['x', 'y', 'z']].values
                print(f"Using explicit x,y,z columns as fallback, loaded {len(pts)} points")
                return pts
            except Exception:
                pass
        return None
        
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        print(f"Exception details: {str(e)}")
        return None

def visualize_point_cloud(points):
    """
    Visualize the point cloud using Open3D
    """
    if points is None or len(points) == 0:
        print("No valid points to visualize")
        return
        
    # Create and visualize point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # Add colors for better visualization (optional)
    colors = np.ones_like(points) * [0.5, 0.5, 0.5]  # Grey color
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Visualize
    print(f"Visualizing {len(points)} points...")
    o3d.visualization.draw_geometries([pcd])

def main():
    # Check if a parquet file path is provided as an argument
    import sys
    if len(sys.argv) < 2:
        print("Please provide the path to the parquet file as an argument")
        print("Usage: python dataset_read.py <path_to_parquet_file> [timestamp_micros]")
        return

    parquet_path = sys.argv[1]
    if not os.path.exists(parquet_path):
        print(f"Error: File {parquet_path} does not exist!")
        return

    # Check if a specific timestamp was requested
    target_timestamp = None
    if len(sys.argv) > 2:
        target_timestamp = int(sys.argv[2])

    print(f"Reading parquet file: {parquet_path}")
    points = read_and_process_lidar(parquet_path, target_timestamp)
    
    if points is not None:
        visualize_point_cloud(points)
    else:
        print("Failed to process point cloud data")

if __name__ == "__main__":
    main()
