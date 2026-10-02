"""Render calibrated mobile-style captures and paired restoration examples.

Uses reviewed full-sphere panoramas; cannot simulate translation parallax.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from panorama.stitching.capture import CameraFrame, Capture, load_capture
from panorama.pano_classical.pose_stitcher import stitch_pose_files


def default_cameras(width=640, height=480, fov=70.):
    f = width / (2 * np.tan(np.deg2rad(fov) / 2))
    cameras = []
    angles = [(yaw, pitch) for pitch in (-45, 0, 45) for yaw in range(0, 360, 45)] + [(0, -90), (0, 90)]
    for i, (yaw, pitch) in enumerate(angles):
        y, p = np.deg2rad([yaw, pitch])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        cameras.append(CameraFrame(str(i), f'images/{i:03}.png', width, height,
                                    [f, f, (width-1)/2, (height-1)/2], (ry @ rx).tolist(), [0., 0., 0.]))
    return cameras


def render(panorama, camera):
    """OpenGL camera-to-world rays, periodic longitude, pixel-centre sampling."""
    fx, fy, cx, cy = camera.intrinsics
    x, y = np.meshgrid(np.arange(camera.width, dtype=np.float32), np.arange(camera.height, dtype=np.float32))
    rays = np.stack(((x-cx)/fx, -(y-cy)/fy, -np.ones_like(x)), -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    world = rays @ np.asarray(camera.rotation).T
    h, w = panorama.shape[:2]
    u = ((np.arctan2(world[..., 0], -world[..., 2]) / (2*np.pi) + .5) * w - .5) % w
    v = np.clip((.5 - np.arcsin(np.clip(world[..., 1], -1, 1))/np.pi)*h - .5, 0, h-1)
    return cv2.remap(panorama, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def degrade(image, rng):
    # Deliberately bounded synthetic variation; not a measured phone sensor model.
    exposure = float(rng.uniform(-.25, .25))
    sigma = float(rng.uniform(0, 1.1))
    noise = float(rng.uniform(0, 3))
    quality = int(rng.integers(75, 96))
    result = image.astype(np.float32) * (2 ** exposure)
    if sigma > .05:
        result = cv2.GaussianBlur(result, (0, 0), sigma)
    result = np.clip(result + rng.normal(0, noise, result.shape), 0, 255).astype(np.uint8)
    ok, encoded = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise OSError('JPEG augmentation failed')
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR), dict(exposure_ev=exposure, blur_sigma=sigma, noise_std=noise, jpeg_quality=quality)


def prepare(manifest_path, output, variants=1, seed=42, output_width=1024, frame_width=640, frame_height=480, template=None):
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    spec = json.loads(manifest_path.read_text(encoding='utf-8'))
    if spec.get('version') != 1 or not spec.get('scenes'):
        raise ValueError('Expected version=1 and nonempty scenes list')
    if variants < 1 or output_width < 64 or output_width % 2 or min(frame_width, frame_height) < 16:
        raise ValueError('Invalid rendering dimensions or variant count')
    groups = {}
    rows = []
    # Validate every input before writing; split by physical scene BEFORE rendering.
    for row in spec['scenes']:
        scene_id = row['scene_id']
        group = row.get('source_scene_id', scene_id)
        if not isinstance(scene_id, str) or not scene_id or not all(c.isalnum() or c in '_-' for c in scene_id):
            raise ValueError('scene_id must be a safe nonempty folder name')
        if row.get('split') not in ('train', 'val') or row.get('reviewed') is not True or row.get('projection') != 'equirectangular':
            raise ValueError('Each reference needs reviewed=true, equirectangular projection and train/val split')
        if group in groups and groups[group] != row['split']:
            raise ValueError('Physical scene appears in different splits')
        groups[group] = row['split']
        path = (manifest_path.parent / row['panorama'].replace('\\', '/')).resolve()
        with Image.open(path) as im:
            if im.width != 2*im.height or im.getexif().get(274, 1) != 1:
                raise ValueError('References must be orientation-normalized full-sphere 2:1 panoramas')
        rows.append((row, path))
    if len({r['scene_id'] for r, _ in rows}) != len(rows):
        raise ValueError('Duplicate scene IDs')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose a new output directory; existing datasets are preserved')
    cameras = default_cameras(frame_width, frame_height)
    if template:
        cameras = load_capture(template).frames
        if any(f.projection != 'pinhole' for f in cameras):
            raise ValueError('Mobile template must use pinhole cameras')
        if any(abs(f.width/f.height - frame_width/frame_height) > .01 for f in cameras):
            raise ValueError('Rendered frame aspect ratio must match the mobile template; e.g. 640x360 for 1920x1080')
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    report = dict(domain='synthetic_mobile', seed=seed, template=str(template) if template else None,
                  limitations=['rotation-only rendering; no translation parallax, rolling shutter or moving people'], pairs=[])
    for row, path in rows:
        with Image.open(path) as im:
            panorama = np.array(im.convert('RGB'))[..., ::-1].copy()
        target = cv2.resize(panorama, (output_width, output_width//2), interpolation=cv2.INTER_AREA)
        for variant in range(variants):
            name = f'{row["scene_id"]}_v{variant:03}'
            capture_dir = output / 'captures' / row['split'] / name
            (capture_dir / 'images').mkdir(parents=True)
            generated, settings = [], []
            for i, original in enumerate(cameras):
                # Scale intrinsics along with frame dimensions using pixel centres.
                sx, sy = frame_width/original.width, frame_height/original.height
                fx, fy, cx, cy = original.intrinsics
                frame = CameraFrame(str(i), f'images/{i:03}.png', frame_width, frame_height,
                                     [fx*sx, fy*sy, (cx+.5)*sx-.5, (cy+.5)*sy-.5], original.rotation, [0., 0., 0.])
                image, params = degrade(render(panorama, frame), rng)
                if not cv2.imwrite(str(capture_dir / frame.filename), image):
                    raise OSError('Could not save generated frame')
                generated.append(frame)
                settings.append(params)
            capture = Capture(capture_dir, generated, 'synthetic_mobile_rotation_only')
            capture.save(capture_dir / 'capture.json')
            pair_dir = output / 'restoration' / row['split'] / name
            pair_dir.mkdir(parents=True)
            before = stitch_pose_files([capture.image_path(f) for f in generated], pair_dir / 'before.png',
                       (output_width, output_width//2), capture_dir / 'capture.json',
                       dict(refine_poses=False, recover_capture_headings=False, exposure_compensation=True,
                            seam_blending=True, seam_width=max(256, min(1024, output_width)), blend_bands=3))
            if not cv2.imwrite(str(pair_dir / 'after.png'), target):
                raise OSError('Could not save clean target')
            # The clean target is known even at synthetic missing-coverage pixels;
            # generated holes are not evidence that real hidden surfaces can be recovered.
            record = dict(scene_id=name, source_scene_id=row.get('source_scene_id', row['scene_id']),
                          before='before.png', after='after.png', alignment_verified=True,
                          projection='equirectangular', domain='synthetic_mobile',
                          reference=str(path), variant=variant, augmentations=settings)
            (pair_dir / 'pair.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
            report['pairs'].append(dict(scene_id=name, source_scene_id=record['source_scene_id'], split=row['split'],
                                       pair=pair_dir.relative_to(output).as_posix()))
    (output / 'generation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--template', help='Optional real mobile capture: copy camera intrinsics/rotations, never its images')
    parser.add_argument('--variants', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-width', type=int, default=2048)
    parser.add_argument('--frame-width', type=int, default=640)
    parser.add_argument('--frame-height', type=int, default=480)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.out, args.variants, args.seed, args.output_width,
                              args.frame_width, args.frame_height, args.template), indent=2))


if __name__ == '__main__':
    main()
