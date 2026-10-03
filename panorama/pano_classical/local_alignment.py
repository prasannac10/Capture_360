"""Bounded, confidence-filtered local registration on a spherical canvas."""

import cv2
import numpy as np


def remap_local(image, displacement, nearest=False):
    """Displacements are inverse maps in pixels at their stored resolution."""
    h, w = image.shape[:2]
    sh, sw = displacement.shape[:2]
    flow = cv2.resize(displacement, (w, h), interpolation=cv2.INTER_LINEAR)
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(image, (x + flow[..., 0] * w / sw) % w,
                     np.clip(y + flow[..., 1] * h / sh, 0, h - 1),
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)


def estimate_displacement(source, reference, valid, reference_valid, trusted):
    """Accept reversible flow only where it improves textured overlap.

    Smooth supported vectors into a coarse displacement field. Unsupported
    regions tend to zero; large motion and occlusions are not extrapolated.
    """
    h, w = valid.shape
    a = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    # Padding keeps local registration periodic across the longitude seam.
    pad = min(32, w // 4)
    aa = cv2.copyMakeBorder(a, 0, 0, pad, pad, cv2.BORDER_WRAP)
    bb = cv2.copyMakeBorder(b, 0, 0, pad, pad, cv2.BORDER_WRAP)
    forward = cv2.calcOpticalFlowFarneback(aa, bb, None, .5, 3, 21, 4, 7, 1.5, 0)[:, pad:-pad]
    reverse = cv2.calcOpticalFlowFarneback(bb, aa, None, .5, 3, 21, 4, 7, 1.5, 0)[:, pad:-pad]
    back = remap_local(reverse, forward)
    moved = remap_local(b, forward)
    support = ((valid > 0) & (reference_valid > 0) & (trusted > 0)
               & (remap_local(valid, forward, True) > 0))
    support &= np.linalg.norm(forward + back, axis=2) < .75
    support &= np.linalg.norm(forward, axis=2) < 8
    rows = np.arange(h)[:, None] + forward[..., 1]
    support &= (rows >= 0) & (rows < h - 1)
    gradient = cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    support &= gradient > 12
    before = np.abs(a.astype(np.float32) - b)
    after = np.abs(a.astype(np.float32) - moved)
    support &= (after < 18) & (after <= before + 1)
    weight = support.astype(np.float32)
    def smooth(array):
        extended = cv2.copyMakeBorder(array, 0, 0, pad, pad, cv2.BORDER_WRAP)
        return cv2.GaussianBlur(extended, (0, 0), 8)[:, pad:-pad]
    density = smooth(weight)
    field = smooth(forward * weight[..., None]) / np.maximum(density[..., None], .05)
    # Decay at source boundaries to avoid creating new coverage holes.
    interior = cv2.distanceTransform(valid, cv2.DIST_L2, 3)
    field *= np.clip(interior / 16, 0, 1)[..., None]
    # Reject a field that folds or strongly compresses the image grid.
    dy_x, dx_x = np.gradient(field[..., 0])
    dy_y, dx_y = np.gradient(field[..., 1])
    determinant = (1 + dx_x) * (1 + dy_y) - dy_x * dx_y
    # A smoothed field must still improve its trusted validation pixels.
    check = support & (before > 2)
    candidate = remap_local(b, field)
    error = np.abs(a.astype(np.float32) - candidate)
    accepted = bool(check.sum() >= 100 and determinant.min() > .5
                    and determinant.max() < 1.5
                    and np.mean(error[check]) < .95 * np.mean(before[check]))
    report = {'accepted': accepted, 'support_pixels': int(check.sum()),
              'mean_error_before': float(np.mean(before[check])) if check.any() else None,
              'mean_error_after': float(np.mean(error[check])) if check.any() else None}
    return (field if accepted else np.zeros_like(field)), report


def protect_view_centres(fields, masks, qualities, reference_index=0):
    """Keep the anchor fixed and taper deformation away from each view's core."""
    if not 0 <= reference_index < len(fields):
        raise ValueError('reference_index is outside the input views')
    result = []
    for i, field in enumerate(fields):
        h, w = field.shape[:2]
        scores = np.stack([np.where(cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 0,
                                   cv2.resize(q, (w, h), interpolation=cv2.INTER_LINEAR), -1)
                           for m, q in zip(masks, qualities)])
        protected = (np.argmax(scores, axis=0) == i) & (scores[i] >= 0)
        # Also protect the central, high-quality portion even if another view wins ownership.
        peak = float(scores[i].max())
        if peak > 0:
            protected |= scores[i] >= .8 * peak
        if i == reference_index:
            result.append(np.zeros_like(field))
            continue
        # Periodic distance to the protected core gives a smooth side/corner transition.
        pad = min(32, w)
        free = (~protected).astype(np.uint8)
        extended = cv2.copyMakeBorder(free, 0, 0, pad, pad, cv2.BORDER_WRAP)
        distance = cv2.distanceTransform(extended, cv2.DIST_L2, 3)[:, pad:-pad]
        taper = np.clip(distance / 16, 0, 1)
        result.append(field * taper[..., None])
    return result


def align_views(images, masks, qualities, width=1024, reference_index=0):
    """Register to a fixed central-view mosaic, never to an averaged blend."""
    h, w = images[0].shape[:2]
    size = (min(width, w), min(width, w) // 2)
    small = [cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in images]
    valid = [cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) for m in masks]
    best = np.zeros(size[::-1], np.float32)
    owner = np.full(size[::-1], -1, np.int16)
    reference = np.zeros((*size[::-1], 3), np.uint8)
    for i, (im, mask, quality) in enumerate(zip(small, valid, qualities)):
        q = cv2.resize(quality, size, interpolation=cv2.INTER_LINEAR)
        use = (q > best) & (mask > 0)
        reference[use], best[use], owner[use] = im[use], q[use], i
    # Do not fit across discontinuities already present in the reference.
    boundary = np.zeros_like(best, np.uint8)
    boundary[:, 1:] |= (owner[:, 1:] != owner[:, :-1]).astype(np.uint8)
    boundary[1:] |= (owner[1:] != owner[:-1]).astype(np.uint8)
    boundary[:, 0] |= (owner[:, 0] != owner[:, -1]).astype(np.uint8)
    boundary = cv2.dilate(boundary, np.ones((11, 11), np.uint8))
    fields, reports = [], []
    for i, (im, mask) in enumerate(zip(small, valid)):
        if i == reference_index:
            fields.append(np.zeros((*size[::-1], 2), np.float32))
            reports.append({'accepted': False, 'reason': 'fixed_reference_view', 'support_pixels': 0})
            continue
        trusted = ((boundary == 0) & (owner != i)).astype(np.uint8)
        field, report = estimate_displacement(im, reference, mask, (owner >= 0).astype(np.uint8), trusted)
        fields.append(field)
        reports.append(report)
    fields = protect_view_centres(fields, valid, qualities, reference_index)
    return fields, {'method': 'bounded_reference_flow', 'work_size': list(size),
                    'reference_index': reference_index, 'protected_centres': True, 'frames': reports}
