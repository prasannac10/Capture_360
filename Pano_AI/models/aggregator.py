import torch
import torch.nn as nn


class SetAggregator(nn.Module):
    """Permutation-invariant context aggregator over the captured views."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
    def forward(self, feats, mask=None):
        pooled = feats.mean(dim=(-1, -2))
        if mask is None:
            set_mean = pooled.mean(dim=1)
        else:
            weights = mask.to(dtype=pooled.dtype).unsqueeze(-1)
            set_mean = (pooled * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.net(set_mean)
