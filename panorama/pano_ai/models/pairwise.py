"""Independently trained alignment and blending heads on geometric RGB warps."""
import torch
from torch import nn
from .decoder import PanoramaConv2d


class PairwiseHead(nn.Module):
    def __init__(self, task, channels=32, max_displacement=8.):
        super().__init__()
        if task not in ('alignment', 'blending'):
            raise ValueError('Unknown pairwise task')
        self.task, self.max_displacement = task, max_displacement
        self.net = nn.Sequential(PanoramaConv2d(8, channels, 3, padding=1), nn.GELU(),
                                 PanoramaConv2d(channels, channels, 3, padding=1), nn.GELU(),
                                 PanoramaConv2d(channels, 3 if task == 'alignment' else 2, 3, padding=1))

    def forward(self, reference, source, reference_valid, source_valid):
        raw = self.net(torch.cat((reference, source, reference_valid, source_valid), dim=1))
        if self.task == 'alignment':
            return {'flow': raw[:, :2].tanh() * self.max_displacement, 'confidence': raw[:, 2:3].sigmoid()}
        return {'weight': raw[:, :1].sigmoid(), 'confidence': raw[:, 1:2].sigmoid()}


def inverse_warp(source, flow):
    """Same inverse, pixel-centre, longitude-periodic convention as inference."""
    import torch.nn.functional as F
    b, _, h, w = source.shape
    y, x = torch.meshgrid(torch.arange(h, device=source.device, dtype=source.dtype),
                          torch.arange(w, device=source.device, dtype=source.dtype), indexing='ij')
    xx = (x + flow[:, 0]).remainder(w)
    yy = (y + flow[:, 1]).clamp(0, h - 1)
    padded = F.pad(source, (1, 1, 0, 0), mode='circular')
    grid = torch.stack((2 * (xx + 1.5) / (w + 2) - 1, 2 * (yy + .5) / h - 1), -1)
    return F.grid_sample(padded, grid, align_corners=False, padding_mode='border')


def blend_pair(reference, source, reference_valid, source_valid, weight):
    a, b = weight * reference_valid, (1 - weight) * source_valid
    return (a * reference + b * source) / (a + b).clamp_min(1e-6)
