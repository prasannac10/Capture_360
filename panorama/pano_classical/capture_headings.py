"""Detect coherent ARCore yaw-frame jumps in Capture360's guided ring capture."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def wrap_degrees(angle):
    return (angle + 180) % 360 - 180


def recover_capture_headings(paths, records, rotations, manifest_path):
    """Use known guided slot spacing only for coherent, large heading jumps.

    Three or more equator slots must corroborate the same offset. The offset
    is applied as one world-Y rotation, retaining measured pitch, roll and
    within-group yaw differences. Feature refinement still follows this prior.
    """
    original = np.asarray(rotations, dtype=np.float64)
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        return original, {'status': 'no_capture_manifest'}
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    slots = manifest.get('slots', [])
    if manifest.get('source') != 'arcore':
        return original, {'status': 'unsupported_capture_manifest'}
    counts = {'equator': 12, 'down': 10, 'up': 10, 'bottom': 4, 'top': 4}
    expected = {f'{row}_{i:02}': (i - 1) * 360 / count
                for row, count in counts.items() for i in range(1, count + 1)}
    if not set(expected).issubset(slots):
        return original, {'status': 'unsupported_slot_layout'}
    names = [Path(p).stem for p in paths]
    equator = [i for i, name in enumerate(names) if name.startswith('equator_') and name in expected]
    if len(equator) < 6:
        return original, {'status': 'insufficient_equator_views'}
    headings = np.rad2deg(np.arctan2(-original[:, 0, 2], original[:, 2, 2]))
    anchor = min(equator, key=lambda i: records[i].get('ts', i))
    base = headings[anchor] - expected[names[anchor]]
    residual = {i: float(wrap_degrees(headings[i] - base - expected[name]))
                for i, name in enumerate(names) if name in expected}
    suspect = [i for i in equator if 45 < abs(residual[i]) < 135]
    groups = []
    while suspect:
        best = max(([i for i in suspect if abs(wrap_degrees(residual[i] - residual[j])) < 8]
                    for j in suspect), key=len)
        if len(best) < 3:
            break
        groups.append(float(np.median([residual[i] for i in best])))
        suspect = [i for i in suspect if i not in best]
    result = original.copy()
    assignments = {}
    for i, value in residual.items():
        assignments[i] = next((g for g in groups if abs(wrap_degrees(value - g)) < 12), 0)
    # At the poles heading is ill-conditioned. Use the nearest timed guided
    # view instead; a pole image does not independently establish a yaw reset.
    for i, name in enumerate(names):
        if name in ('nadir', 'zenith') and assignments and 'ts' in records[i]:
            timed = [j for j in assignments if 'ts' in records[j]]
            if timed:
                nearest = min(timed, key=lambda j: abs(records[j]['ts'] - records[i]['ts']))
                assignments[i] = assignments[nearest]
    changes = []
    for i, offset in assignments.items():
        if offset:
            # Heading is atan2(-R02, R22); positive world-Y rotation removes
            # a positive heading offset in that convention.
            result[i] = Rotation.from_euler('y', offset, degrees=True).as_matrix() @ original[i]
            changes.append({'frame': names[i], 'yaw_offset_removed_deg': offset})
    return result, {'status': 'recovered' if changes else 'consistent',
                    'basis': 'guided_capture_slot_spacing', 'anchor': names[anchor],
                    'offset_groups_deg': groups, 'corrected_frames': changes}
