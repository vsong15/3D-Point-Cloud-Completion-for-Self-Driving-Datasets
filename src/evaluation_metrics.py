from typing import Tuple
import argparse
import os
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def _pairwise_distances_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=-1)


def chamfer_distance_np(pred: np.ndarray,
                        target: np.ndarray,
                        squared: bool = False) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if pred.size == 0 or target.size == 0:
        raise ValueError("Chamfer distance undefined for empty point clouds.")

    d1 = _pairwise_distances_np(pred, target)
    d2 = _pairwise_distances_np(target, pred)

    m1 = np.min(d1, axis=1)
    m2 = np.min(d2, axis=1)

    cd = np.mean(m1) + np.mean(m2)
    return float(cd if squared else np.sqrt(cd))


def fscore_np(pred: np.ndarray,
              target: np.ndarray,
              tau: float = 0.05) -> Tuple[float, float, float]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    if pred.size == 0 or target.size == 0:
        raise ValueError("F-score undefined for empty point clouds.")

    d1 = _pairwise_distances_np(pred, target)
    d2 = _pairwise_distances_np(target, pred)

    m1 = np.sqrt(np.min(d1, axis=1))
    m2 = np.sqrt(np.min(d2, axis=1))

    p = float(np.mean(m1 < tau))
    r = float(np.mean(m2 < tau))

    f = 0.0 if (p + r == 0) else (2 * p * r) / (p + r)
    return f, p, r


def _ensure_torch():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available.")


def chamfer_distance_torch(pred,
                           target,
                           squared: bool = False,
                           reduction: str = "mean"):
    _ensure_torch()
    import torch

    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)

    if pred.shape[0] != target.shape[0]:
        raise ValueError("Batch size mismatch.")

    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    d = torch.sum(diff * diff, dim=-1)

    m1 = torch.min(d, dim=2)[0]
    m2 = torch.min(d, dim=1)[0]

    cd = torch.mean(m1, dim=1) + torch.mean(m2, dim=1)

    if not squared:
        cd = torch.sqrt(cd + 1e-8)

    if reduction == "none":
        return cd
    if reduction == "mean":
        return cd.mean()
    if reduction == "sum":
        return cd.sum()
    raise ValueError("Invalid reduction type.")


def fscore_torch(pred,
                 target,
                 tau: float = 0.05,
                 reduction: str = "mean"):
    _ensure_torch()
    import torch

    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)

    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    d = torch.sum(d * 0 + diff * diff, dim=-1)  # avoid unused warning for diff? (optional)


def fscore_torch(pred,
                 target,
                 tau: float = 0.05,
                 reduction: str = "mean"):
    _ensure_torch()
    import torch

    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)

    diff = pred.unsqueeze(2) - target.unsqueeze(1)
    d = torch.sum(diff * diff, dim=-1)

    m1 = torch.sqrt(torch.min(d, dim=2)[0] + 1e-8)
    m2 = torch.sqrt(torch.min(d, dim=1)[0] + 1e-8)

    p = (m1 < tau).float().mean(dim=1)
    r = (m2 < tau).float().mean(dim=1)

    f = torch.zeros_like(p)
    mask = (p + r) > 0
    f[mask] = 2 * p[mask] * r[mask] / (p[mask] + r[mask])

    if reduction == "none":
        return f, p, r
    if reduction == "mean":
        return f.mean(), p.mean(), r.mean()
    if reduction == "sum":
        return f.sum(), p.sum(), r.sum()
    raise ValueError("Invalid reduction type.")


def _load_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError("NumPy file must have shape (N, 3).")
        return arr
    if ext in (".ply", ".pcd", ".xyz"):
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Point cloud file must contain (N, 3) points.")
        return pts
    raise ValueError(f"Unsupported file extension: {ext}")


def _demo():
    print("NumPy metrics demo")
    gt = np.array([[0, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=np.float32)
    pred = gt + 0.01

    cd = chamfer_distance_np(pred, gt)
    f, p, r = fscore_np(pred, gt, tau=0.05)

    print(f"Chamfer Distance: {cd:.6f}")
    print(f"F-score: {f:.4f}, Precision: {p:.4f}, Recall: {r:.4f}")

    if TORCH_AVAILABLE:
        print("\nPyTorch metrics demo")
        import torch
        gt_t = torch.from_numpy(gt)
        pred_t = torch.from_numpy(pred)
        cd_t = chamfer_distance_torch(pred_t, gt_t)
        f_t, p_t, r_t = fscore_torch(pred_t, gt_t, tau=0.05)
        print(f"Chamfer Distance (Torch): {cd_t.item():.6f}")
        print(f"F-score (Torch): F={f_t.item():.4f}, P={p_t.item():.4f}, R={r_t.item():.4f}")
    else:
        print("PyTorch not available.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, help="Path to ground truth point cloud (.ply or .npy)")
    parser.add_argument("--pred", type=str, help="Path to predicted point cloud (.ply or .npy)")
    parser.add_argument("--tau", type=float, default=0.05, help="Distance threshold for F-score")
    parser.add_argument("--use_torch", action="store_true", help="Also compute Torch metrics if available")
    args = parser.parse_args()

    if args.gt is None or args.pred is None:
        _demo()
        return

    gt = _load_points(args.gt)
    pred = _load_points(args.pred)

    print("NumPy metrics")
    cd = chamfer_distance_np(pred, gt)
    f, p, r = fscore_np(pred, gt, tau=args.tau)
    print(f"Chamfer Distance: {cd:.6f}")
    print(f"F-score: {f:.4f}, Precision: {p:.4f}, Recall: {r:.4f}")

    if args.use_torch:
        if not TORCH_AVAILABLE:
            print("PyTorch not available, skipping Torch metrics.")
        else:
            import torch
            print("\nPyTorch metrics")
            gt_t = torch.from_numpy(gt)
            pred_t = torch.from_numpy(pred)
            cd_t = chamfer_distance_torch(pred_t, gt_t)
            f_t, p_t, r_t = fscore_torch(pred_t, gt_t, tau=args.tau)
            print(f"Chamfer Distance (Torch): {cd_t.item():.6f}")
            print(f"F-score (Torch): F={f_t.item():.4f}, P={p_t.item():.4f}, R={r_t.item():.4f}")


if __name__ == "__main__":
    main()
