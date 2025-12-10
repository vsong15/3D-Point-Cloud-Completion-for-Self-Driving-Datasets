"""
HOW TO RUN
python evaluation_metrics.py \
    --gt path/to/gt_cloud.npy \
    --pred path/to/pred_cloud.npy

    ground truth vs predicted
"""

from typing import Tuple
import argparse
import os
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from extensions.emd.emd_module import emdModule
    EMD_AVAILABLE = True
except Exception:
    EMD_AVAILABLE = False


def _pairwise_distances_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a[:, None, :] - b[None, :, :]
    return np.sum(diff * diff, axis=-1)


def chamfer_distance_np(pred: np.ndarray, target: np.ndarray, squared: bool = False) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    d1 = _pairwise_distances_np(pred, target)
    d2 = _pairwise_distances_np(target, pred)

    m1 = np.min(d1, axis=1)
    m2 = np.min(d2, axis=1)

    cd = np.mean(m1) + np.mean(m2)
    return float(cd if squared else np.sqrt(cd))


def fscore_np(pred: np.ndarray, target: np.ndarray, tau: float = 0.05) -> Tuple[float, float, float]:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    d1 = _pairwise_distances_np(pred, target)
    d2 = _pairwise_distances_np(target, pred)

    m1 = np.sqrt(np.min(d1, axis=1))
    m2 = np.sqrt(np.min(d2, axis=1))

    p = float(np.mean(m1 < tau))
    r = float(np.mean(m2 < tau))

    f = 0.0 if (p + r == 0) else (2 * p * r) / (p + r)
    return f, p, r


def emd_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    d = _pairwise_distances_np(pred, target)
    assigned = np.min(d, axis=1)
    return float(np.mean(np.sqrt(assigned)))


def chamfer_distance_torch(pred, target, squared: bool = False, reduction: str = "mean"):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available.")

    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)

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


def fscore_torch(pred, target, tau: float = 0.05, reduction: str = "mean"):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available.")

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


def emd_torch(pred, target, eps=0.005, iterations=1000):
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch not available.")

    if not EMD_AVAILABLE:
        pred_np = pred.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()
        return torch.tensor(emd_np(pred_np, target_np))

    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    if target.ndim == 2:
        target = target.unsqueeze(0)

    emd = emdModule()
    dist = emd(pred, target, eps, iterations)
    return dist.mean()


def _load_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        return arr
    if ext in (".ply", ".pcd", ".xyz"):
        import open3d as o3d
        pc = o3d.io.read_point_cloud(path)
        return np.asarray(pc.points, dtype=np.float32)
    raise ValueError(f"Unsupported file type: {ext}")


def _demo():
    gt = np.array([[0, 0, 0],
                   [1, 0, 0],
                   [0, 1, 0]], dtype=np.float32)
    pred = gt + 0.01

    cd = chamfer_distance_np(pred, gt)
    f, p, r = fscore_np(pred, gt)
    emd = emd_np(pred, gt)

    print(f"CD: {cd:.6f}")
    print(f"F-score: {f:.4f}, P={p:.4f}, R={r:.4f}")
    print(f"EMD: {emd:.6f}")

    if TORCH_AVAILABLE:
        t_pred = torch.from_numpy(pred)
        t_gt = torch.from_numpy(gt)
        cd_t = chamfer_distance_torch(t_pred, t_gt)
        f_t, p_t, r_t = fscore_torch(t_pred, t_gt)
        emd_t = emd_torch(t_pred, t_gt)

        print(f"CD (Torch): {cd_t.item():.6f}")
        print(f"F-score (Torch): F={f_t.item():.4f}, P={p_t.item():.4f}, R={r_t.item():.4f}")
        print(f"EMD (Torch): {emd_t.item():.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str)
    parser.add_argument("--pred", type=str)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--use_torch", action="store_true")
    args = parser.parse_args()

    if args.gt is None or args.pred is None:
        _demo()
        return

    gt = _load_points(args.gt)
    pred = _load_points(args.pred)

    print("NumPy Metrics:")
    cd = chamfer_distance_np(pred, gt)
    f, p, r = fscore_np(pred, gt, tau=args.tau)
    emd = emd_np(pred, gt)

    print(f"Chamfer Distance: {cd:.6f}")
    print(f"F-score: {f:.4f}, Precision={p:.4f}, Recall={r:.4f}")
    print(f"EMD: {emd:.6f}")

    if args.use_torch and TORCH_AVAILABLE:
        print("\nTorch Metrics:")
        t_pred = torch.from_numpy(pred)
        t_gt = torch.from_numpy(gt)

        cd_t = chamfer_distance_torch(t_pred, t_gt)
        f_t, p_t, r_t = fscore_torch(t_pred, t_gt)
        emd_t = emd_torch(t_pred, t_gt)

        print(f"Chamfer Distance (Torch): {cd_t.item():.6f}")
        print(f"F-score (Torch): F={f_t.item():.4f}, P={p_t.item():.4f}, R={r_t.item():.4f}")
        print(f"EMD (Torch): {emd_t.item():.6f}")


if __name__ == "__main__":
    main()
