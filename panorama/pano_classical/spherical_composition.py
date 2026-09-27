"""Masked exposure estimation and seam ownership for spherical frames."""

import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def spherical_rays(width):
    lon = ((np.arange(width, dtype=np.float32) + 0.5) / width - 0.5) * (2 * np.pi)
    lat = (0.5 - (np.arange(width // 2, dtype=np.float32) + 0.5) / (width // 2)) * np.pi
    return np.stack(np.broadcast_arrays(np.cos(lat[:, None]) * np.sin(lon),
                    np.sin(lat[:, None]), -np.cos(lat[:, None]) * np.cos(lon)), axis=-1)


def warp_frame(path, record, rotation, rays):
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if frame is None or frame.shape[:2] != (record['h'], record['w']):
        raise ValueError(f'Image dimensions must match ARCore calibration: {path}')
    camera = rays @ np.asarray(rotation, np.float32)
    depth = -camera[..., 2]
    safe_depth = np.maximum(depth, 1e-6)
    u = record['fx'] * camera[..., 0] / safe_depth + record['cx']
    v = -record['fy'] * camera[..., 1] / safe_depth + record['cy']
    edge = np.minimum.reduce([u, v, record['w'] - 1 - u, record['h'] - 1 - v])
    valid = (edge > 0) & (depth > 0)
    # Extend source edge colours for the Laplacian pyramid. Black outside the
    # footprint bleeds into valid pixels at coarse bands and creates halos.
    # The separate mask still strictly limits real coverage.
    u = np.clip(u, 0, record['w'] - 1)
    v = np.clip(v, 0, record['h'] - 1)
    warped = cv2.remap(frame, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    # Continuous centrality for coverage fallback; never infer validity from RGB.
    quality = np.where(valid, np.maximum(depth, 0) * np.clip(edge / 32, 0, 1), 0)
    return warped, valid.astype(np.uint8) * 255, quality


def exposure_gains(images, masks):
    """Robust scalar gains, bounded to avoid amplifying bad overlap estimates."""
    n = len(images)
    gray = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for im in images]
    rows, targets = [], []
    for i in range(n):
        for j in range(i + 1, n):
            valid = ((masks[i] > 0) & (masks[j] > 0) & (gray[i] > 15)
                     & (gray[j] > 15) & (gray[i] < 240) & (gray[j] < 240))
            if valid.sum() < 100:
                continue
            ratios = np.log(gray[j][valid] / gray[i][valid])
            median = np.median(ratios)
            if np.median(np.abs(ratios - median)) > 0.12:
                continue
            row = np.zeros(n)
            row[i], row[j] = 1, -1
            rows.append(row)
            targets.append(median)
    if not rows:
        return np.ones(n)
    # Regularize unconstrained cameras to unity and fix the common gain gauge.
    matrix = np.vstack([rows, np.eye(n) * 0.1])
    target = np.r_[targets, np.zeros(n)]
    gains = np.linalg.lstsq(matrix, target, rcond=None)[0]
    return np.clip(np.exp(gains), 0.8, 1.25)


def seam_masks(images, masks):
    """Real graph-cut ownership, estimated on a padded longitude canvas."""
    if len(images) == 1:
        return [masks[0].copy()]
    pad = min(64, images[0].shape[1] // 4)
    padded_images = [cv2.copyMakeBorder(im.astype(np.float32), 0, 0, pad, pad,
                                      cv2.BORDER_WRAP) for im in images]
    padded_masks = [cv2.copyMakeBorder(mask, 0, 0, pad, pad, cv2.BORDER_WRAP) for mask in masks]
    rectangles = [cv2.boundingRect(mask) for mask in padded_masks]
    active = [i for i, (_, _, w, h) in enumerate(rectangles) if w and h]
    cropped_images, cropped_masks, corners = [], [], []
    for i in active:
        x, y, w, h = rectangles[i]
        cropped_images.append(padded_images[i][y:y + h, x:x + w].copy())
        cropped_masks.append(cv2.UMat(padded_masks[i][y:y + h, x:x + w].copy()))
        corners.append((x, y))
    finder = cv2.detail_GraphCutSeamFinder('COST_COLOR_GRAD')
    if len(active) > 1:
        finder.find(cropped_images, corners, cropped_masks)
    result = [np.zeros_like(mask) for mask in padded_masks]
    for i, mask in zip(active, cropped_masks):
        x, y, w, h = rectangles[i]
        result[i][y:y + h, x:x + w] = mask.get()
    return [mask[:, pad:-pad].copy() for mask in result]


def compose_sphere(paths, records, rotations, width, compensate=True, find_seams=True,
                   local_alignment=False, seam_width=1024, blend_bands=5, source_regions=None):
    """Keep only thumbnails in memory; stream large warps into the blender."""
    if len(paths) == 1:
        image, valid, _ = warp_frame(paths[0], records[0], rotations[0], spherical_rays(width))
        image[valid == 0] = 0
        return image, valid, {'exposure_gains': [1.0], 'seam_method': 'single_frame',
                              'coverage_fallback_pixels': 0}
    seam_width = min(width, seam_width)
    rays = spherical_rays(seam_width)
    small_images, valid_masks, qualities = [], [], []
    for path, record, rotation in zip(paths, records, rotations):
        image, mask, quality = warp_frame(path, record, rotation, rays)
        small_images.append(image)
        valid_masks.append(mask)
        qualities.append(quality)
    gains = exposure_gains(small_images, valid_masks) if compensate else np.ones(len(paths))
    small_images = [np.clip(image * gain, 0, 255).astype(np.uint8)
                    for image, gain in zip(small_images, gains)]
    fields = [None] * len(paths)
    alignment_report = {'method': 'disabled'}
    if local_alignment:
        from .local_alignment import align_views, remap_local
        LOGGER.warning('Estimating bounded local alignment for %d views', len(paths))
        fields, alignment_report = align_views(small_images, valid_masks, qualities)
        LOGGER.warning('Accepted local alignment for %d/%d views',
                       sum(item['accepted'] for item in alignment_report['frames']), len(paths))
        small_images = [remap_local(im, flow) for im, flow in zip(small_images, fields)]
        valid_masks = [remap_local(mask, flow, True) for mask, flow in zip(valid_masks, fields)]
    del qualities
    LOGGER.warning('Selecting seams for %d masked views at %dx%d', len(paths), seam_width, seam_width // 2)
    ownership = seam_masks(small_images, valid_masks) if find_seams else valid_masks
    region_report = []
    if source_regions:
        from .source_regions import preserve_source_regions
        ownership, region_report = preserve_source_regions(ownership, valid_masks, paths, source_regions)
    del small_images, valid_masks
    rays = spherical_rays(width)
    height = width // 2
    pad = min(128, width // 4)
    blender = cv2.detail_MultiBandBlender()
    bands = min(blend_bands, max(1, int(np.log2(max(2, width))) - 5))
    blender.setNumBands(bands)
    blender.prepare((0, 0, width + 2 * pad, height))
    coverage = np.zeros((height, width), np.uint8)
    best_quality = np.zeros((height, width), np.float32)
    fallback = np.zeros((height, width, 3), np.uint8)
    LOGGER.warning('Compositing %d views at %dx%d with multiband blending', len(paths), width, height)
    for path, record, rotation, gain, owned, flow in zip(paths, records, rotations, gains, ownership, fields):
        image, valid, quality = warp_frame(path, record, rotation, rays)
        if flow is not None:
            image = remap_local(image, flow)
            valid = remap_local(valid, flow, True)
            quality = remap_local(quality, flow)
        image = np.clip(image * gain, 0, 255).astype(np.uint8)
        better = quality > best_quality
        fallback[better] = image[better]
        best_quality[better] = quality[better]
        coverage |= valid
        # A narrow overlap permits multiband transitions without averaging
        # entire buildings. Longitude padding makes dilation periodic.
        owned = cv2.copyMakeBorder(owned, 0, 0, 2, 2, cv2.BORDER_WRAP)
        owned = cv2.dilate(owned, np.ones((3, 3), np.uint8))[:, 2:-2]
        mask = cv2.resize(owned, (width, height), interpolation=cv2.INTER_NEAREST) & valid
        image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_WRAP)
        mask = cv2.copyMakeBorder(mask, 0, 0, pad, pad, cv2.BORDER_WRAP)
        blender.feed(image.astype(np.int16), mask, (0, 0))
    result, result_mask = blender.blend(None, None)
    result = np.clip(result[:, pad:pad + width], 0, 255).astype(np.uint8)
    missing = (result_mask[:, pad:pad + width] == 0) & (coverage > 0)
    # At an outer coverage boundary coarse pyramid weights approach zero.
    # Preserve observed pixels there instead of accepting normalization halos.
    distance = cv2.distanceTransform(
        cv2.copyMakeBorder(coverage, 0, 0, pad, pad, cv2.BORDER_WRAP), cv2.DIST_L2, 3
    )[:, pad:pad + width]
    missing |= (coverage > 0) & (distance < 2 ** bands)
    result[missing] = fallback[missing]
    result[coverage == 0] = 0
    return result, coverage, {'exposure_gains': gains.tolist(),
                              'local_alignment': alignment_report,
                              'source_regions': region_report,
                              'seam_width': seam_width, 'blend_bands': bands,
                              'seam_method': 'graph_cut' if find_seams else 'all_valid',
                              'coverage_fallback_pixels': int(missing.sum())}
