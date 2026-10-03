"""Decode RAW scenes without spatial downsampling or destructive cleanup."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from panorama.stitching.capture import Capture, load_capture


def decode_dng(path):
    try:
        import rawpy
    except ImportError as error:
        raise RuntimeError('Install rawpy in the preparation environment: pip install rawpy') from error
    with rawpy.imread(str(path)) as raw:
        # Keep sensor orientation and full active dimensions for calibration.
        return raw.postprocess(half_size=False, output_bps=8, user_flip=0,
                               use_camera_wb=True, no_auto_bright=True,
                               output_color=rawpy.ColorSpace.sRGB)


def prepare(scene, output, capture_path=None, target_size=(12000, 6000), pts_path=None):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    files = sorted((scene / 'images').glob('*'))
    files = [p for p in files if p.is_file() and p.suffix.lower() in {'.dng', '.jpg', '.jpeg'}]
    if not files:
        raise ValueError('Expected full-resolution JPEG or DNG photos in scene/images')
    if len({p.stem.lower() for p in files}) != len(files):
        raise ValueError('Duplicate image stems')
    reference = scene / 'Stitched.jpg'
    with Image.open(reference) as image:
        if image.size != tuple(target_size):
            raise ValueError(f'Require genuine {target_size} Stitched.jpg; target resizing is disabled')
    if output.exists():
        raise FileExistsError('Use a new output directory; existing scenes are never overwritten')
    if capture_path and pts_path:
        raise ValueError('Use either existing capture calibration or PTGui import')
    from .ptgui_calibration import PTGuiCalibration
    pts = PTGuiCalibration(pts_path) if pts_path else None
    if pts and set(pts.records) != {p.stem.lower() for p in files}:
        raise ValueError('PTGui must describe exactly the supplied frames')
    capture = load_capture(Path(capture_path).resolve()) if capture_path else None
    calibrated = {Path(f.filename).stem.lower(): f for f in capture.frames} if capture else {}
    if capture and set(calibrated) != {p.stem.lower() for p in files}:
        raise ValueError('Calibration must describe exactly the supplied frames')
    (output / 'images').mkdir(parents=True)
    rows, frames = [], []
    for path in files:
        is_jpeg = path.suffix.lower() in {'.jpg', '.jpeg'}
        if is_jpeg:
            with Image.open(path) as image:
                if image.format != 'JPEG':
                    raise ValueError(f'Expected JPEG data: {path}')
                array = np.array(image.convert('RGB'), copy=True)
        else:
            array = decode_dng(path)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError('RAW decoder must return uint8 sRGB')
        zoom = None
        if pts:
            array, frame, zoom = pts.rectify(path.stem, array)
        h, w = array.shape[:2]
        filename = f'images/{path.stem}.jpg'
        if capture:
            frame = calibrated[path.stem.lower()]
            if (frame.width, frame.height) != (w, h):
                raise ValueError('Calibration dimensions must match full-resolution image dimensions; do not reuse half-size intrinsics')
            if frame.depth:
                raise ValueError('Depth assets require separate preparation')
            frames.append(replace(frame, filename=filename))
        if pts:
            frames.append(replace(frame, filename=filename))
        if is_jpeg and pts is None:
            # Preserve existing JPEG bytes without another compression pass.
            shutil.copy2(path, output / filename)
        else:
            Image.fromarray(array).save(output / filename, format='JPEG', quality=100, subsampling=0)
        rows.append(dict(source=str(path), filename=filename, width=w, height=h, rectification_zoom=zoom))
    # Preserve reference bytes: converting JPEG to PNG adds no recovered detail.
    shutil.copy2(reference, output / 'Stitched.jpg')
    if (scene / 'Panorama.pts').exists():
        shutil.copy2(scene / 'Panorama.pts', output / 'Panorama.pts')
    if capture or pts:
        Capture(output, frames, capture.source if capture else 'PTGui rectilinear JSON, control-point verified').validate().save(output / 'capture.json')
    report = dict(version=1, half_size=False, rgb_format='JPEG, 8-bit sRGB', jpeg_quality=100, jpeg_subsampling=0,
                  target_size=list(target_size), raw_files_deleted=False, frames=rows,
                  calibration_status='verified' if capture or pts else 'required before training',
                  ready_for_calibration_loading=bool(capture or pts), ptgui_validation=pts.validation if pts else None)
    (output / 'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--capture', help='Existing calibrated capture.json matching full-size decoded pixels')
    parser.add_argument('--pts', help='Optimized JSON Panorama.pts; rectify lenses and generate capture.json')
    args = parser.parse_args()
    print(json.dumps(prepare(args.scene, args.output, args.capture, pts_path=args.pts), indent=2))


if __name__ == '__main__':
    main()
