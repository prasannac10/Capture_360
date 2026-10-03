"""Bounded exposure matching from corresponding calibrated overlap pixels.

No reference panorama is used. Estimate one linear-light gain per frame, keeping
color ratios intact. Unconnected/unreliable views retain unity gain.
"""
import math

import cv2
import numpy as np
import torch
from PIL import Image

from ..models.tile_spherical import project_camera_pixels


DEFAULTS = dict(enabled=True, preview_width=512, min_overlap=64, max_gain=2.0,
                max_log_mad=0.2)


def exposure_settings(config=None):
    values = dict(DEFAULTS)
    if config:
        if set(config) - values.keys():
            raise ValueError('Unknown exposure compensation setting')
        values.update(config)
    if (not isinstance(values['enabled'], bool)
            or not isinstance(values['preview_width'], int) or values['preview_width'] < 32
            or not isinstance(values['min_overlap'], int) or values['min_overlap'] < 4
            or not math.isfinite(values['max_gain']) or values['max_gain'] < 1
            or not math.isfinite(values['max_log_mad']) or values['max_log_mad'] <= 0):
        raise ValueError('Invalid exposure compensation settings')
    return values


def srgb_to_linear(value):
    return np.where(value <= .04045, value / 12.92, ((value + .055) / 1.055) ** 2.4)


def linear_to_srgb(value):
    value = np.clip(value, 0, 1)
    return np.where(value <= .0031308, value * 12.92, 1.055 * value ** (1 / 2.4) - .055)


def apply_exposure(rgb, gain):
    """RGB float pixels in [0,1]; keep unity exactly unchanged."""
    if gain == 1.0:
        return rgb
    return linear_to_srgb(srgb_to_linear(rgb) * gain).astype(np.float32)


def solve_overlap_gains(luminance, valid, settings=None):
    """Robust overlap graph solve with zero mean log gain in each component."""
    cfg = exposure_settings(settings)
    n = len(luminance)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            mask = (valid[i] & valid[j] & np.isfinite(luminance[i]) & np.isfinite(luminance[j])
                    & (luminance[i] > 0) & (luminance[j] > 0))
            if int(mask.sum()) < cfg['min_overlap']:
                continue
            ratio = np.log(luminance[j][mask]) - np.log(luminance[i][mask])
            median = float(np.median(ratio))
            mad = float(np.median(np.abs(ratio - median)))
            if not math.isfinite(median) or mad > cfg['max_log_mad']:
                continue
            # Robust estimates resist moving objects and moderate misregistration.
            weight = min(int(mask.sum()), 4096) / (1 + (mad / .05) ** 2)
            edges.append((i, j, median, weight))
    gains = np.ones(n, dtype=np.float64)
    remaining = set(range(n))
    while remaining:
        component = {min(remaining)}
        while True:
            expanded = component | {j for i, j, _, _ in edges if i in component} | {
                i for i, j, _, _ in edges if j in component}
            if expanded == component:
                break
            component = expanded
        remaining -= component
        indices = sorted(component)
        if len(indices) == 1:
            continue
        lookup = {index: k for k, index in enumerate(indices)}
        selected = [edge for edge in edges if edge[0] in component]
        matrix = np.zeros((len(selected), len(indices)))
        target, confidence = [], []
        for row, (i, j, delta, weight) in enumerate(selected):
            matrix[row, lookup[i]], matrix[row, lookup[j]] = 1, -1
            target.append(delta)
            confidence.append(weight)
        target = np.asarray(target)
        base = np.asarray(confidence) / max(confidence)
        robust = np.ones(len(selected))
        for _ in range(4):
            weight = np.sqrt(base * robust)
            system = np.vstack([matrix * weight[:, None], np.ones((1, len(indices)))])
            solution = np.linalg.lstsq(system, np.r_[target * weight, 0.], rcond=None)[0]
            residual = np.abs(matrix @ solution - target)
            robust = np.minimum(1., .1 / np.maximum(residual, 1e-8))
        gains[indices] = np.exp(np.clip(solution, -math.log(cfg['max_gain']), math.log(cfg['max_gain'])))
    before = [abs(delta) for _, _, delta, _ in edges]
    after = [abs(math.log(gains[i] / gains[j]) - delta) for i, j, delta, _ in edges]
    # A badly inconsistent overlap graph must not degrade the measured mismatch.
    if after and np.mean(after) > np.mean(before):
        gains[:] = 1
        after = before
    return gains.tolist(), dict(overlap_pairs=len(edges),
                               mean_log_error_before=float(np.mean(before)) if before else None,
                               mean_log_error_after=float(np.mean(after)) if after else None)


def estimate_exposure(sample, settings=None):
    cfg = exposure_settings(settings)
    if not cfg['enabled']:
        return dict(enabled=False, gains=[1.] * len(sample['frame_paths']), overlap_pairs=0)
    width = cfg['preview_width']
    luminance, masks = [], []
    with torch.no_grad():
        # Project one camera at a time to bound memory for large capture sets.
        for index, path in enumerate(sample['frame_paths']):
            with Image.open(path) as image:
                image = image.convert('RGB')
                original_w, original_h = image.size
                image.thumbnail((width, width), Image.Resampling.LANCZOS)
                rgb = np.asarray(image, dtype=np.float32) / 255.
            h, w = rgb.shape[:2]
            u, v, valid, edge = project_camera_pixels(
                sample['image_size'][index:index+1][None].float(),
                sample['camera_params'][index:index+1][None].float(),
                sample['poses'][index:index+1][None].float(), width // 2, width)
            mx = ((u[0, 0].numpy() + .5) * w / original_w - .5).astype(np.float32)
            my = ((v[0, 0].numpy() + .5) * h / original_h - .5).astype(np.float32)
            projected = cv2.remap(rgb, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            linear = srgb_to_linear(projected)
            luminance.append(linear @ np.array([.2126, .7152, .0722], np.float32))
            masks.append(valid[0, 0].numpy() & (edge[0, 0, 0].numpy() > .2)
                         & (projected.min(-1) > .03) & (projected.max(-1) < .97))
    gains, report = solve_overlap_gains(luminance, masks, cfg)
    return dict(enabled=True, gains=gains, **report)
