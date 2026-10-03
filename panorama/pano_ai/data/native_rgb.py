"""Native RGB skip inputs for neural decoding; no classical stitcher is used."""
from collections import OrderedDict

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

from ..models.tile_spherical import project_camera_pixels, ypr_to_rot


class NativeRGBSource:
    """Sample original photos onto output tiles, keeping two CPU photos cached.

    Winner ownership avoids double images from averaging translated views.
    Overlap disagreement is passed to the learned head, rather than erased by
    a broad blend. Geometry remains rotation-only and requires calibrated poses.
    """
    def __init__(self, sample, cache_frames=2, coverage_size=None):
        self.sample = sample
        self.cache_frames = max(1, cache_frames)
        self.cache = OrderedDict()
        self.coverage_map = torch.zeros(coverage_size, dtype=torch.bool) if coverage_size else None

    def _image(self, index):
        if index not in self.cache:
            with Image.open(self.sample['frame_paths'][index]) as image:
                array = np.array(image.convert('RGB'), copy=True)
            w, h = self.sample['image_size'][index].int().tolist()
            if array.shape[:2] != (h, w):
                raise ValueError('Native RGB pixels must match camera calibration')
            self.cache[index] = torch.from_numpy(array).permute(2, 0, 1)
            while len(self.cache) > self.cache_frames:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    @torch.no_grad()
    def __call__(self, region, output_size, device):
        top, left, height, width = region
        best = torch.zeros(1, 1, height, width, device=device)
        rgb = torch.zeros(1, 3, height, width, device=device)
        total = torch.zeros_like(best)
        summed, squared = torch.zeros_like(rgb), torch.zeros_like(rgb)
        poses = self.sample['poses'].to(device).float()
        if poses.ndim == 2:
            poses = ypr_to_rot(poses)
        sizes = self.sample['image_size'].to(device).float()
        cameras = self.sample['camera_params'].to(device).float()
        gains = self.sample.get('exposure', {}).get('gains', [1.] * len(sizes))
        with torch.autocast(device_type=device.type, enabled=False):
            for index in range(len(sizes)):
                u, v, valid, confidence = project_camera_pixels(
                    sizes[index:index + 1][None], cameras[index:index + 1][None],
                    poses[index:index + 1][None], *output_size, region=region)
                if not bool(valid.any()):
                    continue
                # Transfer only the source rectangle needed for this tile.
                uu, vv = u[valid], v[valid]
                w, h = sizes[index].int().tolist()
                x0, y0 = max(0, int(uu.min()) - 1), max(0, int(vv.min()) - 1)
                x1 = min(w, int(uu.max()) + 3)
                y1 = min(h, int(vv.max()) + 3)
                source = self._image(index)[:, y0:y1, x0:x1].to(device).float()[None] / 255.
                gain = float(gains[index])
                if gain != 1.:
                    linear = torch.where(source <= .04045, source / 12.92,
                                         ((source + .055) / 1.055).pow(2.4))
                    linear = (linear * gain).clamp(0, 1)
                    source = torch.where(linear <= .0031308, linear * 12.92,
                                         1.055 * linear.pow(1 / 2.4) - .055)
                gx = 2 * (u[:, 0] - x0 + .5) / (x1 - x0) - 1
                gy = 2 * (v[:, 0] - y0 + .5) / (y1 - y0) - 1
                color = F.grid_sample(source, torch.stack((gx, gy), -1),
                                      align_corners=False, padding_mode='border')
                weight = confidence[:, 0]
                choose = weight > best
                rgb = torch.where(choose, color, rgb)
                best = torch.maximum(best, weight)
                total += weight
                summed += color * weight
                squared += color.square() * weight
            mean = summed / total.clamp_min(1e-12)
            disagreement = (squared / total.clamp_min(1e-12) - mean.square()).clamp_min(0).mean(1, keepdim=True).sqrt()
        coverage = best > 0
        if self.coverage_map is not None:
            oh, ow = output_size
            y0, x0 = max(0, top), max(0, left)
            y1, x1 = min(oh, top + height), min(ow, left + width)
            if y1 > y0 and x1 > x0:
                self.coverage_map[y0:y1, x0:x1] |= coverage[0, 0, y0-top:y1-top, x0-left:x1-left].cpu()
        return rgb, coverage.float(), disagreement
