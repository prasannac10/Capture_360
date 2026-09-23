"""Shared lightweight U-Net backbone used by correction stages."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class RestorationUNet(nn.Module):
    """Small residual U-Net with explicit skip connections.

    The model predicts a correction residual rather than replacing the input.
    ``mask_channels`` allows inpainting stages to provide a validity/correction
    mask alongside RGB data.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 32,
        mask_channels: int = 0,
    ):
        super().__init__()
        total_in = in_channels + mask_channels
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.enc1 = ConvBlock(total_in, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c3, c3)
        self.dec3 = ConvBlock(c3 + c3, c2)
        self.dec2 = ConvBlock(c2 + c2, c1)
        self.dec1 = ConvBlock(c1 + c1, c1)
        self.head = nn.Conv2d(c1, out_channels, 3, padding=1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x, size=ref.shape[-2:], mode="bilinear", align_corners=False
        )

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if mask is not None:
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            x = torch.cat([x, mask.to(dtype=x.dtype)], dim=1)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self._up(b, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        residual = self.head(d1)
        return torch.clamp(x[:, :3] + residual, 0.0, 1.0)
