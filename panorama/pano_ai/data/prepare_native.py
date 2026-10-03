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


def prepare(scene, output, capture_path=None, target_size=(12000, 6000)):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    files = sorted((scene / 'images').glob('*'))
    files = [p for p in files if p.is_file() and p.suffix.lower() == '.dng']
    if not files:
        raise ValueError('Expected DNG photos in scene/images')
    if len({p.stem.lower() for p in files}) != len(files):
        raise ValueError('Duplicate image stems')
    reference = scene / 'Stitched.jpg'
    with Image.open(reference) as image:
        if image.size != tuple(target_size):
            raise ValueError(f'Require genuine {target_size} Stitched.jpg; target resizing is disabled')
    if output.exists():
        raise FileExistsError('Use a new output directory; existing scenes are never overwritten')
    capture = load_capture(Path(capture_path).resolve()) if capture_path else None
    calibrated = {Path(f.filename).stem.lower(): f for f in capture.frames} if capture else {}
    if capture and set(calibrated) != {p.stem.lower() for p in files}:
        raise ValueError('Calibration must describe exactly the supplied RAW frames')
    (output / 'images').mkdir(parents=True)
    rows, frames = [], []
    for path in files:
        array = decode_dng(path)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
            raise ValueError('RAW decoder must return uint8 sRGB')
        h, w = array.shape[:2]
        filename = f'images/{path.stem}.jpg'
        if capture:
            frame = calibrated[path.stem.lower()]
            if (frame.width, frame.height) != (w, h):
                raise ValueError('Calibration dimensions must match full-resolution RAW decoding; do not reuse half-size intrinsics')
            if frame.depth:
                raise ValueError('Depth assets require separate preparation')
            frames.append(replace(frame, filename=filename))
        Image.fromarray(array).save(output / filename, format='JPEG', quality=100, subsampling=0)
        rows.append(dict(source=str(path), filename=filename, width=w, height=h))
    # Preserve reference bytes: converting JPEG to PNG adds no recovered detail.
    shutil.copy2(reference, output / 'Stitched.jpg')
    if (scene / 'Panorama.pts').exists():
        shutil.copy2(scene / 'Panorama.pts', output / 'Panorama.pts')
    if capture:
        Capture(output, frames, capture.source).validate().save(output / 'capture.json')
    report = dict(version=1, half_size=False, rgb_format='JPEG, 8-bit sRGB', jpeg_quality=100, jpeg_subsampling=0,
                  target_size=list(target_size), raw_files_deleted=False, frames=rows,
                  calibration_status='verified' if capture else 'required before training',
                  ready_for_calibration_loading=bool(capture))
    (output / 'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--capture', help='Existing calibrated capture.json matching full-size decoded pixels')
    args = parser.parse_args()
    print(json.dumps(prepare(args.scene, args.output, args.capture), indent=2))


if __name__ == '__main__':
    main()
