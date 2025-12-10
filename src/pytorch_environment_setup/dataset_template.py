import os, glob, numpy as np, torch
from torch.utils.data import Dataset, DataLoader

class PointCloudDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        if not self.files:
            raise FileNotFoundError(f"No .npy files in {data_dir}")
        self.transform = transform

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        pc = np.load(self.files[idx])  # (N, 3) or (N, C)
        if self.transform: pc = self.transform(pc)
        if isinstance(pc, np.ndarray): pc = torch.from_numpy(pc)
        return pc.float()

def collate_sample_to_fixed(batch, N=1024):
    out = []
    for pc in batch:
        n = pc.shape[0]
        idx = torch.randperm(n)[:N] if n >= N else torch.randint(0, n, (N,))
        out.append(pc[idx])
    return torch.stack(out, dim=0)  # (B, N, C)

def main():
    import argparse, torch.nn as nn
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="mock_data")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    ds = PointCloudDataset(args.data_dir)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.num_workers, collate_fn=collate_sample_to_fixed)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print("Using device:", device)

    first = next(iter(dl))                     # (B, 1024, 3)
    C = first.shape[-1]
    mlp = nn.Sequential(nn.Linear(C, 32), nn.ReLU(), nn.Linear(32, 16)).to(device)

    for i, batch in enumerate(dl):
        batch = batch.to(device)
        out = mlp(batch)                       # (B, 1024, 16)
        print(f"Batch {i}: {tuple(batch.shape)} -> out {tuple(out.shape)}")
        if i == 2: break

if __name__ == "__main__":
    main()
