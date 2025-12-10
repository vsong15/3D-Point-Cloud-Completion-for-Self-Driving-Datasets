import torch
from torch.utils.data import DataLoader
from dataset_template import ObjectPointCloudDataset
from src.baseline_model import BaselineAE
from evaluation_metrics import chamfer_distance

device = "mps"  # or "cuda"

def main():
    dataset = ObjectPointCloudDataset(root="objects/", augment=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = BaselineAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(25):
        for partial, complete in loader:
            partial, complete = partial.to(device), complete.to(device)

            pred = model(partial)
            loss = chamfer_distance(pred, complete)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch} — Chamfer Loss: {loss.item():.4f}")

if __name__ == "__main__":
    main()
