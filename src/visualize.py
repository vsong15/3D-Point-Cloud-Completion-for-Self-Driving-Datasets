import os
import open3d as o3d
import numpy as np

GROUND_TRUTH_FOLDER = "waymo_final_data_splits_updated_normalized/val/completed"
OCCLUDED_FOLDER = "waymo_final_data_splits_updated_normalized/val/incomplete"
COMPLETED_FOLDER = "inference_result_non_fine_tuned_incomplete_ply"

def load_pc(path):
    pc = o3d.io.read_point_cloud(path)
    if not pc.has_points():
        raise FileNotFoundError(f"Could not load point cloud from {path} or it is empty.")
    return pc

def normalize_unit_sphere(pc):
    pts = np.asarray(pc.points)
    center = pts.mean(axis=0)
    pts = pts - center
    scale = np.linalg.norm(pts, axis=1).max()
    if scale > 0:
        pts = pts / scale
    pc.points = o3d.utility.Vector3dVector(pts)
    return pc

def scale_to_match(target_pts, source_pts):
    target_radius = np.linalg.norm(target_pts, axis=1).max()
    source_center = source_pts.mean(axis=0)
    source_pts_centered = source_pts - source_center
    source_radius = np.linalg.norm(source_pts_centered, axis=1).max()
    if source_radius > 0:
        scale_factor = target_radius / source_radius
        source_pts_scaled = source_pts_centered * scale_factor + source_center
        return source_pts_scaled
    return source_pts

def visualize_three_way(gt_path, occluded_path, completed_path, point_size=5, side_offset=3.0):
    gt = load_pc(gt_path)
    occluded = load_pc(occluded_path)
    completed = load_pc(completed_path)

    occluded = normalize_unit_sphere(occluded)
    occluded_pts = np.asarray(occluded.points)

    gt_pts = np.asarray(gt.points)

    completed_pts = np.asarray(completed.points)
    completed_pts = scale_to_match(gt_pts, completed_pts)

    occluded_pts += np.array([-side_offset, 0, 0])
    gt_pts += np.array([0, 0, 0])
    completed_pts += np.array([side_offset, 0, 0])

    occluded.points = o3d.utility.Vector3dVector(occluded_pts)
    gt.points = o3d.utility.Vector3dVector(gt_pts)
    completed.points = o3d.utility.Vector3dVector(completed_pts)

    occluded.paint_uniform_color([1.0, 0.0, 0.0])
    gt.paint_uniform_color([0.0, 1.0, 0.0])
    completed.paint_uniform_color([0.0, 0.0, 1.0])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Ground Truth | Occluded | Completed", width=1200, height=600)
    vis.add_geometry(gt)
    vis.add_geometry(occluded)
    vis.add_geometry(completed)

    render_opt = vis.get_render_option()
    render_opt.point_size = point_size
    render_opt.background_color = np.array([1.0, 1.0, 1.0])
    render_opt.show_coordinate_frame = False

    vis.run()
    vis.destroy_window()

def visualize_folders(gt_folder, occluded_folder, completed_folder):
    gt_files = sorted([f for f in os.listdir(gt_folder) if f.lower().endswith(('.ply', '.pcd'))])
    occluded_files = sorted([f for f in os.listdir(occluded_folder) if f.lower().endswith(('.ply', '.pcd'))])
    completed_files = sorted([f for f in os.listdir(completed_folder) if f.lower().endswith(('.ply', '.pcd'))])

    for gt_file, occluded_file, completed_file in zip(gt_files, occluded_files, completed_files):
        visualize_three_way(
            os.path.join(gt_folder, gt_file),
            os.path.join(occluded_folder, occluded_file),
            os.path.join(completed_folder, completed_file)
        )

if __name__ == "__main__":
    visualize_folders(GROUND_TRUTH_FOLDER, OCCLUDED_FOLDER, COMPLETED_FOLDER)
