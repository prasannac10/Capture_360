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
        detail_mode='features',
        residual_scale=0.1,
    ):
        super().__init__()
        if output_tile <= 0 or not 0 <= output_overlap < output_tile:
            raise ValueError('Require output_tile > output_overlap >= 0')
        self.output_size = output_size
        self.output_tile = output_tile
        self.output_overlap = output_overlap
        if detail_mode not in ('features', 'rgb_residual'):
            raise ValueError('Unknown panorama detail mode')
        if not 0 < residual_scale <= 1:
            raise ValueError('residual_scale must be in (0, 1]')
        self.detail_mode, self.residual_scale = detail_mode, residual_scale
        self.in_proj = nn.Sequential(PanoramaConv2d(dim, dim, 3, padding=1), nn.GELU())
        self.refine = nn.Sequential(
            *[ResidualBlock(dim) for _ in range(refinement_blocks)]
        )
        self.detail = nn.Sequential(
            nn.Conv2d(dim + (5 if detail_mode == 'rgb_residual' else 0), dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, output_channels, 3, padding=1),
        )
        if detail_mode == 'rgb_residual':
            # Start as an exact native RGB skip; learn corrections, not texture
            # reconstruction from an upsampled feature canvas.
            nn.init.zeros_(self.detail[-1].weight)
            nn.init.zeros_(self.detail[-1].bias)

    def forward(self, x, source_sampler=None, region=None):
        return self.forward_tiled(x, self.output_size, source_sampler, region)

    def forward_tiled(self, x, size, source_sampler=None, region=None):
        if self.detail_mode == 'rgb_residual' and source_sampler is None:
            raise ValueError('rgb_residual requires native source RGB sampling')
        x = self.refine(self.in_proj(x))
        oh, ow = size
        th = self.output_tile
        ov = self.output_overlap
        step = th - ov
        if oh <= 0 or ow <= 0:
            raise ValueError('Output dimensions must be positive')
        top, left, rh, rw = region if region is not None else (0, 0, oh, ow)
        if min(top, left) < 0 or min(rh, rw) <= 0 or top + rh > oh or left + rw > ow:
            raise ValueError('Output crop must lie inside the panorama')
        out = x.new_zeros(x.shape[0], self.detail[-1].out_channels, rh, rw, dtype=torch.float32)
        wt = x.new_zeros(x.shape[0], 1, rh, rw, dtype=torch.float32)
        # Global output coordinates ensure tiles sample exactly the same lattice.
        # Two high-resolution halo pixels cover both detail convolutions.
        padded = F.pad(x.float(), (1, 1, 0, 0), mode='circular')
        fh, fw = x.shape[-2:]
        for y in range(top, top + rh, step):
            for xx in range(left, left + rw, step):
                y1, x1 = min(top + rh, y + th), min(left + rw, xx + th)
                rows = torch.arange(y - 2, y1 + 2, device=x.device, dtype=torch.float32)
                columns = torch.arange(xx - 2, x1 + 2, device=x.device, dtype=torch.float32)
                iy = 2 * (rows + .5) / oh - 1
                ix = ((columns + .5) * fw / ow - .5).remainder(fw)
                ix = 2 * (ix + 1.5) / (fw + 2) - 1
                gy, gx = torch.meshgrid(iy, ix, indexing='ij')
                grid = torch.stack((gx, gy), -1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
                with torch.autocast(device_type=x.device.type, enabled=False):
                    high = F.grid_sample(padded, grid, align_corners=False, padding_mode='border').to(x.dtype)
                if self.detail_mode == 'rgb_residual':
                    rgb, coverage, disagreement = source_sampler(
                        (y - 2, xx - 2, y1 - y + 4, x1 - xx + 4), size, x.device)
                    inputs = torch.cat((high, rgb.to(high.dtype), coverage.to(high.dtype),
                                        disagreement.to(high.dtype)), 1)
                    residual = self.residual_scale * torch.tanh(self.detail(inputs).float())
                    pred = ((rgb + residual).clamp(0, 1) * coverage)[:, :, 2:-2, 2:-2]
                else:
                    pred = torch.sigmoid(self.detail(high))[:, :, 2:-2, 2:-2]
                out[:, :, y - top:y1 - top, xx - left:x1 - left] += pred
                wt[:, :, y - top:y1 - top, xx - left:x1 - left] += 1
        return out / wt.clamp_min(1e-6)
