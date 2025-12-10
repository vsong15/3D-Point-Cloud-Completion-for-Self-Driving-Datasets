import os
import numpy as np
from typing import Tuple

GT_FOLDER = "waymo_final_data_splits_updated_normalized/val/completed_npy"
PRED_FOLDER = "inference_result_non_fine_tuned_incomplete"

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

def mse_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if pred.shape != target.shape:
        min_points = min(len(pred), len(target))
        pred = pred[:min_points]
        target = target[:min_points]
    return float(np.mean(np.sum((pred - target) ** 2, axis=1)))

def load_gt_folder(gt_folder: str):
    gt_dict = {}
    for f in os.listdir(gt_folder):
        if f.lower().endswith(".npy"):
            name = os.path.splitext(f)[0]
            gt_dict[name] = os.path.join(gt_folder, f)
    return gt_dict

def load_pred_folder(pred_folder: str):
    pred_dict = {}
    for sub in os.listdir(pred_folder):
        sub_path = os.path.join(pred_folder, sub)
        if not os.path.isdir(sub_path):
            continue
        npy_path = os.path.join(sub_path, "fine.npy")
        if os.path.isfile(npy_path):
            pred_dict[sub] = npy_path
        else:
            print(f"Warning: {sub} missing fine.npy → skipped")
    return pred_dict

def main():
    tau = 0.05
    gt_map = load_gt_folder(GT_FOLDER)
    pred_map = load_pred_folder(PRED_FOLDER)

    print(f"\nFound {len(gt_map)} GT files")
    print(f"Found {len(pred_map)} predicted folders\n")

    common_ids = sorted(list(set(gt_map.keys()) & set(pred_map.keys())))
    print(f"Evaluating {len(common_ids)} matching samples...\n")

    all_cd, all_f, all_p, all_r, all_emd, all_mse = [], [], [], [], [], []

    for id_ in common_ids:
        gt = np.load(gt_map[id_]).astype(np.float32)
        pred = np.load(pred_map[id_]).astype(np.float32)

        cd = chamfer_distance_np(pred, gt)
        f, p, r = fscore_np(pred, gt, tau=tau)
        emd = emd_np(pred, gt)
        mse = mse_np(pred, gt)

        all_cd.append(cd)
        all_f.append(f)
        all_p.append(p)
        all_r.append(r)
        all_emd.append(emd)
        all_mse.append(mse)

        print(f"{id_}:  CD={cd:.6f}  F={f:.4f}  P={p:.4f}  R={r:.4f}  EMD={emd:.6f}  MSE={mse:.6f}")

    print("\n================== SUMMARY ==================")
    print(f"Mean Chamfer Distance: {np.mean(all_cd):.6f}")
    print(f"Mean F-score:          {np.mean(all_f):.4f}")
    print(f"Mean Precision:        {np.mean(all_p):.4f}")
    print(f"Mean Recall:           {np.mean(all_r):.4f}")
    print(f"Mean EMD:              {np.mean(all_emd):.6f}")
    print(f"Mean MSE:              {np.mean(all_mse):.6f}")
    print("=============================================")

if __name__ == "__main__":
    main()
