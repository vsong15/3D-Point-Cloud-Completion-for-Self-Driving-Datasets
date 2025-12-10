import os
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

GROUND_TRUTH_FOLDER = "waymo_final_data_splits_updated/train"
INCOMPLETE_FOLDER = os.path.join(GROUND_TRUTH_FOLDER, "incomplete").replace("\\", "/")
COMPLETED_FOLDER = os.path.join(GROUND_TRUTH_FOLDER, "completed").replace("\\", "/")

def visualize_side_by_side(incomplete_path, complete_path):
    incomplete_pc = o3d.io.read_point_cloud(incomplete_path)
    complete_pc = o3d.io.read_point_cloud(complete_path)
    
    if incomplete_pc.is_empty() or complete_pc.is_empty():
        print(f"Warning: {incomplete_path} or {complete_path} is empty.")
        return
    
    incomplete_pc.paint_uniform_color([1, 0, 0])
    
    incomplete_pts = np.asarray(incomplete_pc.points)
    complete_pts = np.asarray(complete_pc.points)
    
    tree = cKDTree(incomplete_pts)
    dists, _ = tree.query(complete_pts, k=1)
    
    occluded_mask = dists > 0.01
    occluded_pc = o3d.geometry.PointCloud()
    occluded_pc.points = o3d.utility.Vector3dVector(complete_pts[occluded_mask])
    occluded_pc.paint_uniform_color([1, 0, 1])
    
    visible_mask = ~occluded_mask
    visible_pc = o3d.geometry.PointCloud()
    visible_pc.points = o3d.utility.Vector3dVector(complete_pts[visible_mask])
    visible_pc.paint_uniform_color([0, 1, 0])
    
    translate_dist = 0.8
    incomplete_pc.translate((-translate_dist, 0, 0))
    visible_pc.translate((translate_dist, 0, 0))
    occluded_pc.translate((translate_dist, 0, 0.05))
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=os.path.basename(incomplete_path) + " | " + os.path.basename(complete_path),
        width=1600, height=800
    )
    vis.add_geometry(incomplete_pc)
    vis.add_geometry(visible_pc)
    vis.add_geometry(occluded_pc)
    
    render_opt = vis.get_render_option()
    render_opt.point_size = 5.0
    render_opt.background_color = np.array([1, 1, 1])
    
    vis.run()
    vis.destroy_window()

def visualize_folder(incomplete_folder, complete_folder):
    incomplete_files = sorted([f for f in os.listdir(incomplete_folder) if f.lower().endswith(".ply")])
    complete_files = sorted([f for f in os.listdir(complete_folder) if f.lower().endswith(".ply")])
    
    for inc_file, comp_file in zip(incomplete_files, complete_files):
        inc_path = os.path.join(incomplete_folder, inc_file)
        comp_path = os.path.join(complete_folder, comp_file)
        visualize_side_by_side(inc_path, comp_path)

if __name__ == "__main__":
    visualize_folder(INCOMPLETE_FOLDER, COMPLETED_FOLDER)
