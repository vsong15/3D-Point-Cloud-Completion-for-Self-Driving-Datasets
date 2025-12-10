import torch
import torch.nn as nn

class BaselineAE(nn.Module):
    def __init__(self, latent_dim=256, out_points=2048):
        super().__init__()
        self.out_points = out_points

        # Encoder: PointNet-style
        self.encoder = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # Decoder: MLP generating 2048 × 3 points
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, out_points * 3)
        )

    def forward(self, x):
        # x: (B, N, 3)
        feats = self.encoder(x)
        global_feat = feats.max(1)[0]  # (B, latent_dim)
        out = self.decoder(global_feat)
        return out.view(-1, self.out_points, 3)
