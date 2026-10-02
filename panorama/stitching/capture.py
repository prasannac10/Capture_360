"""Shared calibrated capture contract. Canonical poses use OpenGL camera axes.

camera-to-world rotation, translation in metres; camera +X right,+Y up,-Z forward.
Image coordinates are original sensor pixels, +X right,+Y down. No EXIF rotation.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import math

import numpy as np

CV_BASIS = np.diag([1., -1., -1.])
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.webp', '.tif', '.tiff'}


@dataclass
class CameraFrame:
    name: str
    filename: str
    width: int
    height: int
    intrinsics: list
    rotation: list
    translation: list
    projection: str = 'pinhole'
    fov_degrees: float = 0.
    depth: dict | None = None
    timestamp: float | None = None
    translation_known: bool = True

    def validate(self):
        r, t, k = np.asarray(self.rotation), np.asarray(self.translation), np.asarray(self.intrinsics)
        if (r.shape != (3, 3) or t.shape != (3,) or k.shape != (4,)
                or not np.isfinite(r).all() or not np.isfinite(t).all() or not np.isfinite(k).all()
                or not np.allclose(r.T @ r, np.eye(3), atol=.01)
                or not np.isclose(np.linalg.det(r), 1, atol=.01)
                or min(self.width, self.height, k[0], k[1]) <= 0):
            raise ValueError(f'Invalid camera calibration: {self.name}')
        if self.projection not in ('pinhole', 'fisheye_180'):
            raise ValueError(f'Unsupported projection: {self.projection}')
        if self.projection == 'fisheye_180' and not 0 < self.fov_degrees <= 180:
            raise ValueError('Fisheye needs a valid fov_degrees')
        if self.depth is not None:
            if not self.depth.get('path'):
                raise ValueError('Depth metadata needs a path')
            # Raw mobile depth remains opaque until its format/calibration is known.
            if self.depth.get('format') == 'npy_meters' and self.depth.get('units') != 'metres':
                raise ValueError('npy_meters depth must declare units=metres')
        return self

    def classical_record(self):
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = self.rotation, self.translation
        record = dict(name=Path(self.filename).stem, m=m.flatten(order='F').tolist(),
                      fx=self.intrinsics[0], fy=self.intrinsics[1], cx=self.intrinsics[2], cy=self.intrinsics[3],
                      w=self.width, h=self.height, projection=self.projection,
                      fov_degrees=self.fov_degrees,
                      depth=self.depth, translation_known=self.translation_known)
        if self.timestamp is not None:
            record['ts'] = self.timestamp
        return record


@dataclass
class Capture:
    root: Path
    frames: list[CameraFrame]
    source: str

    def validate(self):
        if not self.frames or len({f.name for f in self.frames}) != len(self.frames):
            raise ValueError('Capture must have distinct named frames')
        if len({f.filename for f in self.frames}) != len(self.frames):
            raise ValueError('Duplicate capture filenames')
        for f in self.frames:
            f.validate()
        return self

    def image_path(self, frame):
        return (self.root / frame.filename.replace('\\', '/')).resolve()

    def save(self, path):
        path = Path(path).resolve()
        # Write paths relative to destination so the contract can move with its scene.
        import os
        rows = []
        for frame in self.frames:
            row = asdict(frame)
            row['filename'] = os.path.relpath(self.image_path(frame), path.parent).replace('\\', '/')
            if frame.depth:
                row['depth']['path'] = os.path.relpath(self.root / frame.depth['path'], path.parent).replace('\\', '/')
                if frame.depth.get('confidence_path'):
                    row['depth']['confidence_path'] = os.path.relpath(self.root / frame.depth['confidence_path'], path.parent).replace('\\', '/')
            rows.append(row)
        path.write_text(json.dumps({'version': 1, 'coordinates': 'opengl_camera_to_world',
                                    'source': self.source, 'frames': rows}, indent=2), encoding='utf-8')


def _arcore(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    images = {}
    names = {row['name'] for row in rows}
    for p in path.parent.iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and p.stem in names:
            if p.stem in images:
                raise ValueError(f'Duplicate source image stem: {p.stem}')
            images[p.stem] = p.name
    frames = []
    for row in rows:
        m = np.asarray(row['m'], dtype=float).reshape(4, 4, order='F')
        if not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1], atol=.01):
            raise ValueError('Invalid homogeneous ARCore transform')
        name = row['name']
        depth = path.parent / (name + '.depth')
        frames.append(CameraFrame(name, images.get(name, name + '.jpg'), row['w'], row['h'],
                                  [row[k] for k in ('fx', 'fy', 'cx', 'cy')], m[:3, :3].tolist(),
                                  m[:3, 3].tolist(), depth={'path': depth.name, 'format': 'opaque_arcore'} if depth.exists() else None,
                                  timestamp=row.get('ts')))
    return Capture(path.parent, frames, 'arcore').validate()


def _legacy(scene):
    import torch
    from PIL import Image
    from panorama.pano_ai.models.tile_spherical import ypr_to_rot
    files = sorted(p for p in (scene / 'images').iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    state = torch.load(scene / 'poses.pt', map_location='cpu', weights_only=False)
    if isinstance(state, dict) and 'frame_names' in state:
        by_name = {p.name: p for p in files}
        try:
            files = [by_name[name] for name in state['frame_names']]
        except KeyError as error:
            raise ValueError(f'Missing named source image: {error}') from error
    else:
        files = [p for p in files if not any(word in p.stem.lower() for word in ('panorama', 'preview', 'coverage', 'mask'))
                 and p.stem not in ('00_object_removal', '04_color', '09_sharpen')]
    poses = next((state[k] for k in ('poses', 'ypr', 'rotations') if isinstance(state, dict) and k in state), state)
    poses = torch.as_tensor(poses, dtype=torch.float32)
    if poses.shape == (len(files), 3):
        poses = ypr_to_rot(poses)
    if poses.shape != (len(files), 3, 3):
        raise ValueError('Legacy poses must match the ordered source frames')
    camera = json.loads((scene / 'camera.json').read_text(encoding='utf-8'))
    frames = []
    for p, rotation in zip(files, poses.numpy()):
        with Image.open(p) as image:
            w, h = image.size
        typ = camera.get('projection', camera.get('model', 'fisheye_180')).lower().replace('-', '_')
        fisheye = typ in ('fisheye', 'fisheye_180', 'equidistant', '180_degree_equidistant')
        fov = float(camera.get('fov_deg', camera.get('horizontal_fov_deg', 180))) if fisheye else 0.
        if all(k in camera for k in ('fx', 'fy', 'cx', 'cy')):
            k = [float(camera[key]) for key in ('fx', 'fy', 'cx', 'cy')]
        elif fisheye:
            f = min(w, h) / math.radians(fov)
            k = [f, f, w / 2, h / 2]
        elif 'horizontal_fov_deg' in camera:
            f = w / (2 * math.tan(math.radians(camera['horizontal_fov_deg']) / 2))
            k = [f, f, w / 2, h / 2]
        else:
            raise ValueError('Legacy camera.json requires calibrated intrinsics or FOV')
        frames.append(CameraFrame(p.stem, str(p.relative_to(scene)), w, h, k,
                                  (CV_BASIS @ rotation @ CV_BASIS).tolist(), [0., 0., 0.],
                                  'fisheye_180' if fisheye else 'pinhole', fov, translation_known=False))
    return Capture(scene, frames, 'legacy_angles').validate()


def load_capture(path):
    path = Path(path).resolve()
    if path.is_dir():
        for candidate in (path / 'capture.json', path / 'images' / 'ar_poses.jsonl', path / 'ar_poses.jsonl'):
            if candidate.is_file():
                return load_capture(candidate)
        return _legacy(path)
    if path.suffix == '.jsonl':
        return _arcore(path)
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('version') != 1 or data.get('coordinates') != 'opengl_camera_to_world':
        raise ValueError('Unsupported capture version or coordinate system')
    for row in data['frames']:
        row['filename'] = row['filename'].replace('\\', '/')
        if row.get('depth'):
            for key in ('path', 'confidence_path'):
                if key in row['depth']:
                    row['depth'][key] = row['depth'][key].replace('\\', '/')
    return Capture(path.parent, [CameraFrame(**row) for row in data['frames']], data.get('source', 'calibrated')).validate()
