import os, sys
print("Python:", sys.version)

# --- Torch checks ---
import torch
print("torch:", torch.__version__)
print("CUDA available?:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Quick tensor test on CPU/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(4, 3, device=device)
w = torch.randn(3, 2, device=device, requires_grad=True)
y = x @ w
loss = y.pow(2).mean()
loss.backward()
print("Torch small op OK on", device)

# --- PyG checks ---
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

print("torch_geometric imported OK.")

# Tiny graph (just to exercise a layer)
edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)  # 3 nodes, 4 edges (undirected+)
x = torch.randn(3, 8)  # 3 nodes, 8 features
data = Data(x=x, edge_index=edge_index)

conv = GCNConv(in_channels=8, out_channels=4).to(device)
out = conv(data.x.to(device), data.edge_index.to(device))
print("PyG GCNConv forward OK, out shape:", tuple(out.shape))

# Optional: tiny "point cloud" sanity (N x 3)
pc = torch.randn(1024, 3, device=device)
mlp = torch.nn.Sequential(
    torch.nn.Linear(3, 32),
    torch.nn.ReLU(),
    torch.nn.Linear(32, 16),
).to(device)
pc_out = mlp(pc)
print("Point-cloud MLP forward OK, out shape:", tuple(pc_out.shape))

print("\nAll good ✅")
