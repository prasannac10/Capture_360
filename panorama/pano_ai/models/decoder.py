"""Memory-bounded panorama decoder with tiled high-resolution output."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PanoramaConv2d(nn.Conv2d):
    """Same parameters as Conv2d, with periodic longitude and clamped poles."""
    def forward(self, x):
        py, px = self.padding
        if px:
            x = F.pad(x, (px, px, 0, 0), mode='circular')
        if py:
            x = F.pad(x, (0, 0, py, py), mode='replicate')
        return F.conv2d(x, self.weight, self.bias, self.stride, 0, self.dilation, self.groups)


class ResidualBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.net = nn.Sequential(
            PanoramaConv2d(c, c, 3, padding=1), nn.GELU(), PanoramaConv2d(c, c, 3, padding=1)
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
        if output_tile <= 0 or not 0 <= output_overlap < output_tile:
            raise ValueError('Require output_tile > output_overlap >= 0')
        self.output_size = output_size
        self.output_tile = output_tile
        self.output_overlap = output_overlap
        self.in_proj = nn.Sequential(PanoramaConv2d(dim, dim, 3, padding=1), nn.GELU())
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
        if oh <= 0 or ow <= 0:
            raise ValueError('Output dimensions must be positive')
        out = x.new_zeros(x.shape[0], self.detail[-1].out_channels, oh, ow, dtype=torch.float32)
        wt = x.new_zeros(x.shape[0], 1, oh, ow, dtype=torch.float32)
        # Global output coordinates ensure tiles sample exactly the same lattice.
        # Two high-resolution halo pixels cover both detail convolutions.
        padded = F.pad(x.float(), (1, 1, 0, 0), mode='circular')
        fh, fw = x.shape[-2:]
        for y in range(0, oh, step):
            for xx in range(0, ow, step):
                y1, x1 = min(oh, y + th), min(ow, xx + th)
                rows = torch.arange(y - 2, y1 + 2, device=x.device, dtype=torch.float32)
                columns = torch.arange(xx - 2, x1 + 2, device=x.device, dtype=torch.float32)
                iy = 2 * (rows + .5) / oh - 1
                ix = ((columns + .5) * fw / ow - .5).remainder(fw)
                ix = 2 * (ix + 1.5) / (fw + 2) - 1
                gy, gx = torch.meshgrid(iy, ix, indexing='ij')
                grid = torch.stack((gx, gy), -1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
                with torch.autocast(device_type=x.device.type, enabled=False):
                    high = F.grid_sample(padded, grid, align_corners=False, padding_mode='border').to(x.dtype)
                pred = torch.sigmoid(self.detail(high))[:, :, 2:-2, 2:-2]
                out[:, :, y:y1, xx:x1] += pred
                wt[:, :, y:y1, xx:x1] += 1
        return out / wt.clamp_min(1e-6)
