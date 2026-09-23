"""Memory-bounded panorama decoder with tiled high-resolution output."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1), nn.GELU(), nn.Conv2d(c, c, 3, padding=1)
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


class MultiScalePanoramaDecoder(nn.Module):
    def __init__(
        self,
        dim=64,
        output_channels=3,
        output_size=(6000, 12000),
        refinement_blocks=3,
        output_tile=1024,
        output_overlap=64,
    ):
        super().__init__()
        self.output_size = output_size
        self.output_tile = output_tile
        self.output_overlap = output_overlap
        self.in_proj = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.GELU())
        self.refine = nn.Sequential(
            *[ResidualBlock(dim) for _ in range(refinement_blocks)]
        )
        self.detail = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, output_channels, 3, padding=1),
        )

    def forward(self, x):
        return self.forward_tiled(x, self.output_size)

    def forward_tiled(self, x, size):
        x = self.refine(self.in_proj(x))
        oh, ow = size
        th = self.output_tile
        ov = self.output_overlap
        step = th - ov
        out = x.new_zeros(x.shape[0], 3, oh, ow)
        wt = x.new_zeros(x.shape[0], 1, oh, ow)
        for y in list(range(0, max(1, oh - th + 1), step)) + (
            [max(0, oh - th)] if oh > th else []
        ):
            for xx in list(range(0, max(1, ow - th + 1), step)) + (
                [max(0, ow - th)] if ow > th else []
            ):
                y = min(y, max(0, oh - th))
                xx = min(xx, max(0, ow - th))
                y1 = min(oh, y + th)
                x1 = min(ow, xx + th)
                # Map output tile bounds to a slightly expanded low-res feature crop.
                fy0 = max(0, int(y * x.shape[-2] / oh) - 1)
                fx0 = max(0, int(xx * x.shape[-1] / ow) - 1)
                fy1 = min(x.shape[-2], int((y1) * x.shape[-2] / oh) + 2)
                fx1 = min(x.shape[-1], int((x1) * x.shape[-1] / ow) + 2)
                low = x[:, :, fy0:fy1, fx0:fx1]
                high = F.interpolate(
                    low,
                    size=(y1 - y + 2, x1 - xx + 2),
                    mode="bilinear",
                    align_corners=False,
                )
                pred = torch.sigmoid(self.detail(high))[:, :, 1:-1, 1:-1]
                out[:, :, y:y1, xx:x1] += pred[:, :, : y1 - y, : x1 - xx]
                wt[:, :, y:y1, xx:x1] += 1
        return out / wt.clamp_min(1e-6)
