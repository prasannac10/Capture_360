"""Remove explicitly marked objects using nearby texture, without a detector."""

import cv2
import numpy as np


def remove_configured_objects(image, settings):
    """Clone clean donor texture into normalized polygon regions.

    Polygon coordinates and donor offsets are fractions of image width/height.
    Each clone operates on a small crop at native resolution. Pixels outside
    the configured masks are preserved exactly. Donors wrap in longitude.
    """
    regions = settings.get('regions', [])
    if not regions:
        raise ValueError('object_removal is enabled but no regions are configured')
    height, width = image.shape[:2]
    masks = []
    offsets = []
    union = np.zeros((height, width), np.uint8)
    for region in regions:
        points = np.asarray(region.get('polygon', []), dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 3
                or not np.isfinite(points).all() or np.any((points < 0) | (points > 1))):
            raise ValueError('Object polygons need at least three normalized [x, y] points in [0, 1]')
        offset = np.asarray(region.get('donor_offset', []), dtype=np.float64)
        if offset.shape != (2,) or not np.isfinite(offset).all() or np.any(np.abs(offset) > 1):
            raise ValueError('Object donor_offset must be two finite fractions in [-1, 1]')
        polygon = np.rint(points * [width - 1, height - 1]).astype(np.int32)
        x, y, w, h = cv2.boundingRect(polygon)
        if w < 3 or h < 3 or abs(cv2.contourArea(polygon)) < 4:
            raise ValueError('Object polygon is too small or degenerate at this resolution')
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [polygon - [x, y]], 255)
        union[y:y + h, x:x + w] |= mask
        masks.append((x, y, w, h, mask))
        offsets.append(np.rint(offset * [width, height]).astype(int))
    fraction = float(np.mean(union > 0))
    limit = float(settings.get('max_area_fraction', 0.01))
    if not np.isfinite(limit) or not 0 < limit <= 0.1:
        raise ValueError('object_removal.max_area_fraction must be in (0, 0.1]')
    if fraction > limit:
        raise ValueError(f'Object mask covers {fraction:.2%}, exceeding configured limit {limit:.2%}')
    result = image.copy()
    for (x, y, w, h, region_mask), (dx, dy) in zip(masks, offsets):
        # Include unmasked context around the destination for Poisson blending.
        padding = max(8, round(width / 256))
        x0, y0 = max(0, x - padding), max(0, y - padding)
        x1, y1 = min(width, x + w + padding), min(height, y + h + padding)
        rows = np.arange(y0, y1) + dy
        columns = (np.arange(x0, x1) + dx) % width
        if rows.min() < 0 or rows.max() >= height:
            raise ValueError('Object donor region extends past the top or bottom of the panorama')
        local_mask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        local_mask[y - y0:y - y0 + h, x - x0:x - x0 + w] = region_mask
        donor_mask = union[np.ix_(rows, columns)]
        if np.any((donor_mask > 0) & (local_mask > 0)):
            raise ValueError('Object donor overlaps a removal region; choose clean ground in donor_offset')
        donor = image[np.ix_(rows, columns)].copy()
        destination = result[y0:y1, x0:x1].copy()
        # seamlessClone centres the nonzero mask bounding box at this point,
        # not the whole source array (important for asymmetrically clipped ROIs).
        centre = (x - x0 + w // 2, y - y0 + h // 2)
        selected = local_mask > 0
        # OpenCV modifies its mask buffer during cloning. Preserve ownership
        # before that call or only a narrow boundary ring gets copied back.
        blended = cv2.seamlessClone(donor, destination, local_mask.copy(), centre, cv2.NORMAL_CLONE)
        destination[selected] = blended[selected]
        result[y0:y1, x0:x1] = destination
    return result, union, {'method': 'texture_clone', 'regions': len(regions),
                           'mask_fraction': fraction, 'automatic_detection': False}
