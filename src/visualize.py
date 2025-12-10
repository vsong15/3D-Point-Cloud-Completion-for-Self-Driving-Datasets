"""
HOW TO RUN
python visualize_triplet.py \
    --partial waymo_final_dataset_splits_normalized/test/incomplete/0001.ply \
    --pred    predictions/0001.ply \
    --gt      waymo_final_dataset_splits_normalized/test/completed/0001.ply

"""

import open3d as o3d
import numpy as np
import os
import argparse

def load_pc(path):
    pc = o3d.io.read_point_cloud(path)
    pc.paint_uniform_color([0.0, 0.0, 0.0])
    return pc

def visualize_triplet(partial_path, pred_path, gt_path):
    partial = load_pc(partial_path)
    pred = load_pc(pred_path)
    gt = load_pc(gt_path)

    # Move clouds to left, center, right
    partial.translate([-2, 0, 0])
    gt.translate([2, 0, 0])

    o3d.visualization.draw_geometries(
        [partial, pred, gt],
        window_name="Partial (Left) | Predicted (Center) | Ground Truth (Right)"
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--partial", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--gt", required=True)
    args = parser.parse_args()

    visualize_triplet(args.partial, args.pred, args.gt)
