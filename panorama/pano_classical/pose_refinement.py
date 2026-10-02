"""Robust rotation-only refinement with ARCore rotations as soft priors."""

import logging

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial.transform import Rotation

LOGGER = logging.getLogger(__name__)


def pixel_rays(points, record):
    if record.get('projection') == 'fisheye_180':
        xy = np.column_stack(((points[:, 0] - record['cx']) / record['fx'],
                              -(points[:, 1] - record['cy']) / record['fy']))
        theta = np.linalg.norm(xy, axis=1)
        scale = np.sinc(theta / np.pi)
        return np.column_stack((xy * scale[:, None], -np.cos(theta)))
    rays = np.column_stack(((points[:, 0] - record['cx']) / record['fx'],
                            -(points[:, 1] - record['cy']) / record['fy'],
                            -np.ones(len(points))))
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def optimize_rotations(rotations, edges):
    """Edges contain (camera_i, camera_j, corresponding local unit rays).

    Robust loss suppresses residual parallax. Unmatched cameras keep their
    recorded poses. Small priors fix the global gauge without tilting the sky.
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    if not edges:
        return rotations, {'status': 'no_reliable_matches', 'pairs': 0}
    n = len(rotations)
    count = sum(len(a) for _, _, a, _ in edges)
    sparsity = lil_matrix((count * 3 + n * 3, n * 3), dtype=int)
    offset = 0
    for i, j, a, b in edges:
        size = len(a) * 3
        sparsity[offset:offset + size, i * 3:i * 3 + 3] = 1
        sparsity[offset:offset + size, j * 3:j * 3 + 3] = 1
        offset += size
    sparsity[offset:, :] = np.eye(n * 3, dtype=int)

    def matrices(x):
        return Rotation.from_rotvec(x.reshape(n, 3)).as_matrix() @ rotations

    def residual(x):
        current = matrices(x)
        values = [(a @ current[i].T - b @ current[j].T).ravel()
                  for i, j, a, b in edges]
        return np.concatenate(values + [x])

    def robust_data_linear_prior(z):
        # Do not robustify the prior: that would stop penalizing a large tilt.
        root = np.sqrt(1 + z)
        rho = np.stack((2 * (root - 1), 1 / root, -0.5 / root ** 3))
        rho[0, count * 3:] = z[count * 3:]
        rho[1, count * 3:] = 1
        rho[2, count * 3:] = 0
        return rho

    def errors(current):
        return np.concatenate([np.rad2deg(np.arccos(np.clip(
            np.sum((a @ current[i].T) * (b @ current[j].T), axis=1), -1, 1)))
            for i, j, a, b in edges])

    before = errors(rotations)
    fit = least_squares(residual, np.zeros(n * 3), jac_sparsity=sparsity.tocsr(),
                        loss=robust_data_linear_prior, f_scale=0.005, max_nfev=80,
                        bounds=(-np.deg2rad(5), np.deg2rad(5)))
    refined = matrices(fit.x)
    # Matching constrains relative rotations, not the global horizon/heading.
    # Restore the closest common orientation to the ARCore gravity frame.
    connected = sorted({k for i, j, _, _ in edges for k in (i, j)})
    cross = sum(rotations[k] @ refined[k].T for k in connected)
    u, _, vt = np.linalg.svd(cross)
    gauge = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    refined[connected] = gauge @ refined[connected]
    changes = Rotation.from_matrix(refined @ rotations.transpose(0, 2, 1)).magnitude()
    after = errors(refined)
    accepted = bool(np.isfinite(fit.x).all() and np.median(after) < np.median(before)
                    and np.max(changes) < np.deg2rad(8))
    report = {'status': 'refined' if accepted else 'kept_original', 'pairs': len(edges),
              'matches': count, 'median_error_before_deg': float(np.median(before)),
              'median_error_after_deg': float(np.median(after)),
              'p90_error_after_deg': float(np.percentile(after, 90)),
              'correction_degrees': np.rad2deg(changes).tolist()}
    return (refined if accepted else rotations), report


def recover_unmatched_rotations(rotations, raw_edges, trusted):
    """Recover a contradicted pose only when two trusted neighbors agree.

    Use distant/horizon features in the trusted camera, and choose the best
    rotation fit from a consensus. A single apparently good pair is not enough.
    """
    proposals = {}
    for i, j, a, b in raw_edges:
        if i not in trusted and j in trusted:
            i, j, a, b = j, i, b, a
        if i not in trusted or j in trusted:
            continue
        valid = np.abs((a @ rotations[i].T)[:, 1]) < 0.45
        a, b = a[valid], b[valid]
        if len(a) < 12:
            continue
        # Reject tiny feature clusters which poorly constrain a 3D rotation.
        if np.linalg.norm(np.std(a, axis=0)) < 0.04:
            continue
        keep = np.ones(len(a), bool)
        for _ in range(4):
            u, _, vt = np.linalg.svd(a[keep].T @ b[keep])
            relative = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
            error = np.rad2deg(np.arccos(np.clip(np.sum(a * (b @ relative.T), axis=1), -1, 1)))
            keep = error <= max(0.15, np.percentile(error, 75))
        if np.median(error) > 0.6 or np.percentile(error, 90) > 1.5:
            continue
        candidate = rotations[i] @ relative
        angle = np.rad2deg(Rotation.from_matrix(candidate @ rotations[j].T).magnitude())
        if not 3 < angle < 30:
            continue
        proposals.setdefault(j, []).append((i, candidate, float(np.median(error)), len(a)))
    recovered = rotations.copy()
    report = []
    for target, candidates in proposals.items():
        for source, candidate, error, count in sorted(candidates, key=lambda item: item[2]):
            consensus = [other for other in candidates if np.rad2deg(Rotation.from_matrix(
                candidate @ other[1].T).magnitude()) < 5]
            if len(consensus) < 2:
                continue
            recovered[target] = candidate
            report.append({'camera_index': target, 'reference_indices': [p[0] for p in consensus],
                           'median_fit_error_deg': error, 'matches': count,
                           'correction_deg': float(np.rad2deg(Rotation.from_matrix(
                               candidate @ rotations[target].T).magnitude()))})
            break
    return recovered, report


def horizon_edges(raw_edges, rotations):
    edges = []
    for i, j, ra, rb in raw_edges:
        world_a, world_b = ra @ rotations[i].T, rb @ rotations[j].T
        agreement = np.sum(world_a * world_b, axis=1)
        valid = ((agreement > np.cos(np.deg2rad(8)))
                 & (np.abs(world_a[:, 1]) < 0.45)
                 & (np.abs(world_b[:, 1]) < 0.45))
        if valid.sum() < 12:
            continue
        ra, rb = ra[valid], rb[valid]
        idx = np.linspace(0, len(ra) - 1, min(100, len(ra))).astype(int)
        edges.append((i, j, ra[idx], rb[idx]))
    return edges


def refine_capture_rotations(paths, records, rotations):
    detector = cv2.SIFT_create(nfeatures=3000)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    features = []
    for path, record in zip(paths, records):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or image.shape != (record['h'], record['w']):
            raise ValueError(f'Image dimensions must match ARCore calibration: {path}')
        scale = min(1.0, 1000 / max(image.shape))
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        points, descriptors = detector.detectAndCompute(small, None)
        features.append((points, descriptors, scale))
    raw_edges = []
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            # Conservative cone gate; source roll is already in the poses.
            if np.dot(rotations[i][:, 2], rotations[j][:, 2]) < np.cos(np.deg2rad(85)):
                continue
            ka, da, sa = features[i]
            kb, db, sb = features[j]
            if da is None or db is None or min(len(da), len(db)) < 12:
                continue
            matches = [p[0] for p in matcher.knnMatch(da, db, k=2)
                       if len(p) == 2 and p[0].distance < 0.72 * p[1].distance]
            # Avoid many source points voting for one repeated target feature.
            matches = sorted(matches, key=lambda m: m.distance)
            unique = {}
            for match in matches:
                unique.setdefault(match.trainIdx, match)
            matches = list(unique.values())
            if len(matches) < 12:
                continue
            a = np.float32([ka[m.queryIdx].pt for m in matches])
            b = np.float32([kb[m.trainIdx].pt for m in matches])
            _, mask = cv2.findHomography(a, b, cv2.RANSAC, 2.5)
            if mask is None or mask.sum() < 12 or mask.mean() < 0.25:
                continue
            a, b = a[mask.ravel() > 0] / sa, b[mask.ravel() > 0] / sb
            ra, rb = pixel_rays(a, records[i]), pixel_rays(b, records[j])
            raw_edges.append((i, j, ra, rb))
    # Near-ground parallax and moving clouds are poor rotation constraints.
    edges = horizon_edges(raw_edges, rotations)
    refined, report = optimize_rotations(rotations, edges)
    if report['status'] == 'refined':
        trusted = {k for i, j, _, _ in edges for k in (i, j)}
        recovered, recovery_report = recover_unmatched_rotations(refined, raw_edges, trusted)
        if recovery_report:
            for item in recovery_report:
                item['frame'] = paths[item['camera_index']].stem
            refined, second = optimize_rotations(recovered, horizon_edges(raw_edges, recovered))
            report['recovered_poses'] = recovery_report
            report['recovery_refinement'] = second
            report['correction_degrees'] = np.rad2deg(Rotation.from_matrix(
                refined @ np.asarray(rotations).transpose(0, 2, 1)).magnitude()).tolist()
    LOGGER.warning('Camera refinement: %s; %d pairs; median angular error %.3f -> %.3f degrees',
                   report['status'], report.get('pairs', 0),
                   report.get('median_error_before_deg', 0), report.get('median_error_after_deg', 0))
    return refined, report
