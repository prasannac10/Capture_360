import torch
import torch.nn as nn

class SetAggregator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, feats):
        pooled = feats.mean(dim=0)
        return self.net(pooled)