"""Explicit scene-specific seam ownership for difficult static structures."""

from pathlib import Path

import cv2
import numpy as np


def preserve_source_regions(ownership, validity, paths, regions):
    """Keep a structure in one observed source; never extend its coverage."""
    names = [Path(path).stem for path in paths]
    result = [mask.copy() for mask in ownership]
    h, w = validity[0].shape
    reports = []
    for region in regions:
        name = region.get('source')
        if name not in names:
            raise ValueError(f'Structure source is not an input frame: {name}')
        points = np.asarray(region.get('polygon', []), dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or np.any((points < 0) | (points > 1))):
            raise ValueError('Structure polygon requires normalized finite [x,y] points')
        polygon = np.rint(points * [w - 1, h - 1]).astype(np.int32)
        if abs(cv2.contourArea(polygon)) < 4:
            raise ValueError('Structure polygon is degenerate or too small')
        selected = np.zeros((h, w), np.uint8)
        cv2.fillPoly(selected, [polygon], 255)
        index = names.index(name)
        use = (selected > 0) & (validity[index] > 0)
        for i, mask in enumerate(result):
            mask[use] = 255 if i == index else 0
        reports.append({'source': name, 'selected_pixels': int(use.sum()),
                        'observed_fraction': float(use.sum() / np.count_nonzero(selected))})
    return result, reports
