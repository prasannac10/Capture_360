"""Spatial panorama reconstruction/refinement decoder."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.block(x))


class PanoramaDecoder(nn.Module):
    """Decode projected spherical features into a full-resolution RGB panorama."""

    def __init__(self, dim: int, output_channels: int = 3):
        super().__init__()
        hidden = max(32, dim // 2)
        self.in_proj = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
        )
        self.refine = nn.Sequential(ResidualBlock(hidden), ResidualBlock(hidden))
        self.out = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, output_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.refine(self.in_proj(x)))
