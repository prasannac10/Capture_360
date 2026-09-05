import torch.nn as nn


class ImageEncoder(nn.Module):
    """CNN encoder that deliberately preserves spatial information."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, dim, 3, 2, 1), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)
