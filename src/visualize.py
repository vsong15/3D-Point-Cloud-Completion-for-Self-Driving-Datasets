"""
HOW TO RUN
python visualize_pair.py \
  --occluded waymo_final_dataset_splits_normalized/test/incomplete/0001.ply \
  --completed predictions/0001.ply

"""

import open3d as o3d
import argparse


def load_pc(path, color=None):
    pc = o3d.io.read_point_cloud(path)
    if color is not None:
        pc.paint_uniform_color(color)
    return pc


def visualize_pair(occluded_path, completed_path):
    occluded = load_pc(occluded_path, color=[1.0, 0.0, 0.0])   # red
    completed = load_pc(completed_path, color=[0.0, 1.0, 0.0]) # green

    occluded.translate([-1.5, 0.0, 0.0])
    completed.translate([1.5, 0.0, 0.0])

    o3d.visualization.draw_geometries(
        [occluded, completed],
        window_name="Occluded Input (Left) | Completed Output (Right)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--occluded", required=True,
                        help="Path to occluded / partial input point cloud (.ply or .pcd)")
    parser.add_argument("--completed", required=True,
                        help="Path to completed model output point cloud (.ply or .pcd)")
    args = parser.parse_args()

    visualize_pair(args.occluded, args.completed)
