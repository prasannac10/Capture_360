"""Explicit semantic defect masks, separate from valid captured floor/ceiling."""
import json
from pathlib import Path
import cv2
import numpy as np

KINDS = ('missing_coverage', 'moving_objects', 'photographer', 'shadow', 'glare', 'lens_dots', 'other')


def load_defect_masks(scene, shape, coverage=None):
    scene = Path(scene)
    manifest = scene / 'defects.json'
    paths = {}
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding='utf-8'))
        if (data.get('version') != 1 or data.get('coordinates') != 'equirectangular'
                or data.get('size') != [shape[1], shape[0]]):
            raise ValueError('Defect manifest must declare version 1, equirectangular coordinates and exact output size')
        paths = data.get('masks', {})
        if set(paths) - set(KINDS):
            raise ValueError('Unknown defect category')
    else:
        for name, filename in (('other', 'correction_mask.png'), ('moving_objects', 'ghost_mask.png'),
                                ('photographer', 'photographer_mask.png'), ('shadow', 'shadow_mask.png')):
            if (scene / filename).exists():
                paths[name] = filename
    masks = {}
    for name, filename in paths.items():
        image = cv2.imread(str(scene / filename), cv2.IMREAD_GRAYSCALE)
        if image is None or image.shape != tuple(shape):
            raise ValueError(f'{name}: defect mask must exist and match panorama size')
        masks[name] = image.astype(np.float32) / 255
    if coverage is not None:
        if coverage.shape != tuple(shape):
            raise ValueError('Coverage must match panorama size')
        masks['missing_coverage'] = (coverage == 0).astype(np.float32)
    return masks


def union_masks(masks, names):
    selected = [masks[name] for name in names if name in masks]
    return np.maximum.reduce(selected) if selected else None


def save_defect_masks(output, masks):
    directory = Path(output) / 'defect_masks'
    directory.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, mask in masks.items():
        if not cv2.imwrite(str(directory / (name + '.png')), np.rint(mask * 255).astype(np.uint8)):
            raise OSError('Could not write defect mask')
        report[name] = {'fraction': float(np.mean(mask > 0)), 'path': f'defect_masks/{name}.png'}
    return report
