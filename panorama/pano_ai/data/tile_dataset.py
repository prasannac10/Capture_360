"""Lazy native-resolution scene dataset and chunk loader."""

from pathlib import Path
import json, math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .tiling import tile_specs


class VariableTilePanoramaDataset(Dataset):
    def __init__(
        self,
        root,
        has_gt=True,
        tile_size=1024,
        overlap=128,
        min_frames=4,
        max_frames=30,
        max_tiles_per_frame=None,
        input_config=None,
        profile='auto',
    ):
        self.root = Path(root)
        self.has_gt = has_gt
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.max_tiles_per_frame = max_tiles_per_frame
        self.input_config, self.profile = input_config, profile
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.scenes = sorted(
            p for p in self.root.iterdir() if p.is_dir() and ((p / "images").is_dir() or (p / 'capture.json').is_file())
        )
        if not self.scenes:
            raise RuntimeError(f"No scenes under {self.root}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        from panorama.stitching.capture import load_capture, CV_BASIS
        capture = load_capture(scene)
        files = [capture.image_path(frame) for frame in capture.frames]
        n = len(files)
        if self.input_config is None and not self.min_frames <= n <= self.max_frames:
            raise ValueError(f'{scene}: expected {self.min_frames}..{self.max_frames}, got {n}')
        sizes, cams, specs = [], [], []
        rotations, translations, depths = [], [], []
        for p, frame in zip(files, capture.frames):
            with Image.open(p) as image:
                w, h = image.size
            if (w, h) != (frame.width, frame.height):
                raise ValueError(f'{p}: pixels must match calibration dimensions')
            ss = tile_specs(h, w, self.tile_size, self.overlap)
            if self.max_tiles_per_frame and len(ss) > self.max_tiles_per_frame:
                raise ValueError(f'{p}: tile limit exceeded')
            sizes.append([w, h])
            cams.append(torch.tensor(frame.intrinsics + [1. if frame.projection == 'pinhole' else 0., frame.fov_degrees]))
            rotations.append(CV_BASIS @ np.asarray(frame.rotation) @ CV_BASIS)
            translations.append(CV_BASIS @ np.asarray(frame.translation))
            depths.append(dict(frame.depth, path=str(capture.root / frame.depth['path'])) if frame.depth else None)
            specs.append([(s.x, s.y, s.width, s.height) for s in ss])
        poses = torch.tensor(np.array(rotations), dtype=torch.float32)
        if self.input_config is not None:
            from panorama.stitching.profiles import validate_frame_set
            name, selected = validate_frame_set(sizes, self.input_config, self.profile)
            if not selected['min_frames'] <= n <= selected['max_frames']:
                raise ValueError(f'{name}: frame count {n} is outside its configured limits')
            expected = 0 if selected['projection'] == 'fisheye_180' else 1
            if any(float(cam[4]) != expected for cam in cams):
                raise ValueError(f'{name}: projection does not match profile')
        out = {
            "frame_paths": [str(p) for p in files],
            "tile_specs": specs,
            "image_size": torch.tensor(sizes, dtype=torch.float32),
            "camera_params": torch.stack(cams),
            "poses": poses,
            "translations": torch.tensor(np.array(translations), dtype=torch.float32),
            "depth": depths,
            "capture": capture,
            "scene": scene.name,
        }
        if self.has_gt:
            gp = scene / "panorama.png"
            if not gp.exists():
                raise FileNotFoundError(gp)
            with Image.open(gp) as im:
                out["gt_panorama"] = (
                    torch.from_numpy(np.array(im.convert("RGB"), copy=True))
                    .permute(2, 0, 1)
                    .float()
                    / 255.0
                )
        return out


def iter_tile_batches(sample, tile_size=1024, overlap=128, batch_size=4):
    for fi, path in enumerate(sample["frame_paths"]):
        specs = sample["tile_specs"][fi]
        for st in range(0, len(specs), batch_size):
            chosen = specs[st : st + batch_size]
            tiles = []
            with Image.open(path) as im:
                for x, y, w, h in chosen:
                    crop = im.convert("RGB").crop((x, y, x + w, y + h))
                    ph = tile_size - h
                    pw = tile_size - w
                    if ph or pw:
                        arr = np.asarray(crop)
                        mode = "reflect" if min(arr.shape[:2]) > 1 else "edge"
                        arr = np.pad(arr, ((0, ph), (0, pw), (0, 0)), mode=mode)
                        crop = Image.fromarray(arr)
                    tiles.append(
                        torch.from_numpy(np.array(crop, copy=True)).permute(2, 0, 1).float()
                        / 255.0
                    )
            yield fi, torch.stack(tiles), torch.tensor(
                [[x, y] for x, y, _, _ in chosen], dtype=torch.float32
            ), torch.tensor([[w, h] for _, _, w, h in chosen], dtype=torch.float32)


def tile_collate(batch):
    if len(batch) != 1:
        raise ValueError("Native-resolution tiled scenes require batch_size=1")
    return batch[0]
