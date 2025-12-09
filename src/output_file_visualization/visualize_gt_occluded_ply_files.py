import os
import open3d as o3d
import numpy as np
from scipy.spatial import cKDTree

GROUND_TRUTH_FOLDER = "waymo_final_dataset_splits/train"
INCOMPLETE_FOLDER = os.path.join(GROUND_TRUTH_FOLDER, "incomplete").replace("\\", "/")
COMPLETED_FOLDER = os.path.join(GROUND_TRUTH_FOLDER, "completed").replace("\\", "/")

OCCLUDER_SIZE = np.array([1.8, 4.5, 2.0])
OCCLUDER_DISTANCE = 1.0
OCCLUDER_OFFSET_Y = 0.0
OCCLUDER_OFFSET_Z = 0.0

def add_label(text, position):
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    mesh.translate(position)
    return mesh

def create_occluder_box(pc):
    pts = np.asarray(pc.points)
    if len(pts) == 0:
        return None
    min_bounds = pts.min(axis=0)
    max_bounds = pts.max(axis=0)
    center = (min_bounds + max_bounds) / 2

    occ_min = center + np.array([-OCCLUDER_SIZE[0]/2, max_bounds[1]/2 + OCCLUDER_DISTANCE, -OCCLUDER_SIZE[2]/2])
    occ_max = occ_min + OCCLUDER_SIZE
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=occ_min, max_bound=occ_max)
    bbox.color = (1, 1, 0)  
    return bbox

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
    
    visible_pc = o3d.geometry.PointCloud()
    visible_pc.points = o3d.utility.Vector3dVector(complete_pts[~occluded_mask])
    visible_pc.paint_uniform_color([0, 1, 0])  
    
    incomplete_pc.translate((-3, 0, 0))
    visible_pc.translate((3, 0, 0.2))
    occluded_pc.translate((3, 0, 0.2 + 0.05)) 
    
    coord_incomplete = add_label("Incomplete", (-3, 1.5, 0))
    coord_complete = add_label("Complete", (3, 1.5, 0))
    
    occluder_box = create_occluder_box(complete_pc)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=os.path.basename(incomplete_path) + " | " + os.path.basename(complete_path),
        width=1600, height=800
    )
    vis.add_geometry(incomplete_pc)
    vis.add_geometry(visible_pc)
    vis.add_geometry(occluded_pc)
    vis.add_geometry(coord_incomplete)
    vis.add_geometry(coord_complete)
    if occluder_box is not None:
        vis.add_geometry(occluder_box)
    
    render_opt = vis.get_render_option()
    render_opt.point_size = 5.0
    render_opt.background_color = np.array([0, 0, 0])
    
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
