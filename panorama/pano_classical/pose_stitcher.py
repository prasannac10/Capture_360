"""OpenCV spherical reprojection of ARCore captures (no learned models).

ARCore matrices are column-major camera-to-world transforms, with camera
axes +X right, +Y up, -Z forward. Translation is intentionally ignored:
this baseline assumes rotation about one optical centre.
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def load_pose_records(path):
    from panorama.stitching.capture import load_capture
    capture = load_capture(path)
    records = [frame.classical_record() for frame in capture.frames]
    if len({r['name'] for r in records}) != len(records):
        raise ValueError('Classical calibrated inputs require unique filename stems')
    return {record['name']: record for record in records}


def stitch_pose_files(image_paths, output_path, output_size, poses_path, options=None, view_refiner=None):
    """Refine and seam-blend calibrated views on a 360 x 180 degree canvas.

    Compose at up to 4096 pixels wide to bound memory; larger requested
    outputs are upscaled. Unobserved pixels remain black and a coverage mask
    is saved beside the result. Source pixels must match ARCore intrinsics.
    """
    records = load_pose_records(poses_path)
    paths = list(map(Path, image_paths))
    if not paths or len({p.stem for p in paths}) != len(paths):
        raise ValueError("Provide distinct source frames with recorded poses")
    missing = [p.name for p in paths if p.stem not in records]
    if missing:
        raise ValueError(f"No ARCore pose for inputs: {missing}")
    width, height = map(int, output_size)
    if width <= 0 or height <= 0 or width != 2 * height:
        raise ValueError("Spherical output must have positive 2:1 dimensions")
    work_w = min(width, 4096)
    work_h = work_w // 2
    from .pose_refinement import refine_capture_rotations
    from .spherical_composition import compose_sphere

    options = options or {}
    allowed = {'refine_poses', 'exposure_compensation', 'seam_blending', 'recover_capture_headings', 'local_alignment'}
    numeric = {'seam_width': (256, 4096), 'blend_bands': (1, 8)}
    if set(options) - allowed - set(numeric) - {'source_regions'}:
        raise ValueError('Unknown pose option')
    if 'source_regions' in options and not isinstance(options['source_regions'], list):
        raise ValueError('source_regions must be a list')
    for key, value in options.items():
        if key in allowed and not isinstance(value, bool):
            raise ValueError(f'{key} must be boolean')
        if key in numeric and (type(value) is not int or not numeric[key][0] <= value <= numeric[key][1]
                               or (key == 'seam_width' and value % 2)):
            raise ValueError(f'{key} must be an integer in {numeric[key]} (seam_width must be even)')
    camera_records = [records[path.stem] for path in paths]
    rotations = np.array([np.asarray(record['m']).reshape(4, 4, order='F')[:3, :3]
                          for record in camera_records])
    heading_report = {'status': 'disabled'}
    if options.get('recover_capture_headings', False):
        from .capture_headings import recover_capture_headings
        rotations, heading_report = recover_capture_headings(
            paths, camera_records, rotations, Path(poses_path).with_name('capture-info.json'))
        LOGGER.warning('Capture heading recovery: %s (%d frames)', heading_report['status'],
                       len(heading_report.get('corrected_frames', [])))
    refinement = {'status': 'disabled'}
    if options.get('refine_poses', True) and len(paths) > 1:
        rotations, refinement = refine_capture_rotations(paths, camera_records, rotations)
    panorama, coverage, composition = compose_sphere(
        paths, camera_records, rotations, work_w,
        compensate=options.get('exposure_compensation', True),
        find_seams=options.get('seam_blending', True),
        local_alignment=options.get('local_alignment', False),
        seam_width=options.get('seam_width', 1024),
        blend_bands=options.get('blend_bands', 5),
        source_regions=options.get('source_regions', []), view_refiner=view_refiner)
    covered_fraction = float(np.mean(coverage > 0))
    if (work_w, work_h) != (width, height):
        panorama = cv2.resize(panorama, (width, height), interpolation=cv2.INTER_LANCZOS4)
        coverage = cv2.resize(coverage, (width, height), interpolation=cv2.INTER_NEAREST)
    output_path = Path(output_path)
    for path, data in ((output_path, panorama), (output_path.with_name(output_path.stem + "_coverage.png"), coverage)):
        if not cv2.imwrite(str(path), data):
            raise OSError(f"Could not write {path}")
    output_path.with_name(output_path.stem + "_geometry.json").write_text(json.dumps({
        "method": "arcore_refined_multiband", "frames": len(paths),
        "coverage_fraction": covered_fraction, "work_size": [work_w, work_h],
        "output_size": [width, height], "translation_used": False,
        "pose_refinement": refinement, "composition": composition,
        "capture_heading_recovery": heading_report,
        "frame_names": [path.stem for path in paths],
        "refined_rotations": rotations.tolist(),
    }, indent=2), encoding="utf-8")
    LOGGER.warning("ARCore spherical baseline: %d frames, %.2f%% coverage", len(paths), 100 * covered_fraction)
    return panorama
