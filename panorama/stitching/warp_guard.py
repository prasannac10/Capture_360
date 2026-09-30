"""Observable checks on an inverse warp; uncertainty never becomes geometry."""
import cv2
import numpy as np


def sample(array, flow, nearest=False):
    h, w = array.shape[:2]
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(array, (x + flow[..., 0]) % w, np.clip(y + flow[..., 1], 0, h - 1),
                     cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def guard_warp(reference, source, reference_valid, source_valid, flow, confidence, reverse_flow,
               max_displacement=8., min_confidence=.8, min_pixels=100, max_cycle_error=1.):
    shape = reference.shape[:2]
    fallback = np.zeros((*shape, 2), np.float32)
    report = {'accepted': False, 'fallback': 'geometric_projection'}
    arrays = (flow, confidence, reverse_flow)
    if (flow.shape != (*shape, 2) or confidence.shape != shape or reverse_flow.shape != flow.shape
            or any(not np.isfinite(a).all() for a in arrays)):
        return fallback, dict(report, reason='invalid_prediction')
    if np.any((confidence < 0) | (confidence > 1)) or np.max(np.linalg.norm(flow, axis=-1)) > max_displacement:
        return fallback, dict(report, reason='bounds')
    dy_x, dx_x = np.gradient(flow[..., 0])
    dy_y, dx_y = np.gradient(flow[..., 1])
    determinant = (1 + dx_x) * (1 + dy_y) - dy_x * dx_y
    if determinant.min() < .5 or determinant.max() > 1.5:
        return fallback, dict(report, reason='fold_or_stretch')
    y = np.arange(shape[0])[:, None] + flow[..., 1]
    cycle = np.linalg.norm(flow + sample(reverse_flow, flow), axis=-1)
    support = ((reference_valid > 0) & (source_valid > 0) & (sample(source_valid, flow, True) > 0)
               & (confidence >= min_confidence) & (cycle <= max_cycle_error)
               & (y >= 0) & (y < shape[0] - 1))
    # Textureless agreement is not evidence of a reliable correspondence.
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    evidence = support & (gradient > 12)
    report['supported_pixels'] = int(evidence.sum())
    if evidence.sum() < min_pixels:
        return fallback, dict(report, reason='insufficient_confident_overlap')
    # Taper inside confidence/validity support, not across low-texture holes.
    # The distance ramp keeps unsupported pixels fixed without sharp flow steps.
    radius = max(4., max_displacement * 4)
    pad = min(shape[1], int(np.ceil(radius)) + 2)
    padded = cv2.copyMakeBorder(support.astype(np.uint8), 0, 0, pad, pad, cv2.BORDER_WRAP)
    padded = cv2.copyMakeBorder(padded, 1, 1, 0, 0, cv2.BORDER_CONSTANT, value=0)
    distance = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, pad:-pad]
    strength = np.minimum(distance / radius, 1.) * support
    field = flow * strength[..., None]
    before = np.abs(reference.astype(np.float32) - source).mean(axis=-1)
    after = np.abs(reference.astype(np.float32) - sample(source, field)).mean(axis=-1)
    report.update(error_before=float(before[evidence].mean()), error_after=float(after[evidence].mean()))
    if report['error_after'] >= .95 * report['error_before']:
        return fallback, dict(report, reason='no_photometric_improvement')
    # Recheck the supported field: multiplying confidence can itself cause folds.
    dy_x, dx_x = np.gradient(field[..., 0])
    dy_y, dx_y = np.gradient(field[..., 1])
    jacobian = (1 + dx_x) * (1 + dy_y) - dy_x * dx_y
    if jacobian.min() < .5 or jacobian.max() > 1.5:
        return fallback, dict(report, reason='support_boundary_fold')
    return field, dict(report, accepted=True, fallback=None, reason='accepted')
