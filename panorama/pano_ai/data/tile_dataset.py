"""Lazy native-resolution scene dataset and chunk loader."""

from pathlib import Path
import json, math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .tiling import tile_specs


def _camera(scene, w, h):
    p = scene / "camera.json"
    d = (
        json.loads(p.read_text())
        if p.exists()
        else {"projection": "fisheye_180", "fov_deg": 180.0}
    )
    typ = (
        str(d.get("projection", d.get("model", "fisheye_180")))
        .lower()
        .replace("-", "_")
    )
    if typ in {"fisheye", "fisheye_180", "equidistant", "180_degree_equidistant"}:
        fov = float(d.get("fov_deg", d.get("horizontal_fov_deg", 180.0)))
        fr = math.radians(fov)
        f = min(w, h) / fr
        return torch.tensor(
            [
                float(d.get("fx", f)),
                float(d.get("fy", f)),
                float(d.get("cx", w / 2)),
                float(d.get("cy", h / 2)),
                0.0,
                fov,
            ]
        )
    if all(k in d for k in ("fx", "fy", "cx", "cy")):
        vals = [float(d[k]) for k in ("fx", "fy", "cx", "cy")]
    elif "horizontal_fov_deg" in d:
        hf = math.radians(float(d["horizontal_fov_deg"]))
        fx = w / (2 * math.tan(hf / 2))
        vf = 2 * math.atan((h / w) * math.tan(hf / 2))
        vals = [fx, h / (2 * math.tan(vf / 2)), w / 2, h / 2]
    else:
        raise ValueError(f"{p} needs calibrated intrinsics or horizontal_fov_deg")
    return torch.tensor(vals + [1.0, 0.0])


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
    ):
        self.root = Path(root)
        self.has_gt = has_gt
        self.tile_size = tile_size
        self.overlap = overlap
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.max_tiles_per_frame = max_tiles_per_frame
        if not self.root.exists():
            raise FileNotFoundError(self.root)
        self.scenes = sorted(
            p for p in self.root.iterdir() if p.is_dir() and (p / "images").is_dir()
        )
        if not self.scenes:
            raise RuntimeError(f"No scenes under {self.root}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        files = sorted(
            p
            for p in (scene / "images").iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"}
        )
        n = len(files)
        if not self.min_frames <= n <= self.max_frames:
            raise ValueError(
                f"{scene}: expected {self.min_frames}..{self.max_frames}, got {n}"
            )
        pp = scene / "poses.pt"
        if not pp.exists():
            raise FileNotFoundError(pp)
        poses = torch.load(pp, map_location="cpu", weights_only=False)
        poses = next(
            (
                poses[k]
                for k in ("poses", "ypr", "rotations")
                if isinstance(poses, dict) and k in poses
            ),
            poses,
        )
        poses = torch.as_tensor(poses, dtype=torch.float32)
        if poses.shape != (n, 3):
            raise ValueError(f"{scene}: poses must be [N,3], got {tuple(poses.shape)}")
        sizes = []
        cams = []
        specs = []
        for p in files:
            with Image.open(p) as im:
                w, h = im.size
            ss = tile_specs(h, w, self.tile_size, self.overlap)
            if self.max_tiles_per_frame and len(ss) > self.max_tiles_per_frame:
                raise ValueError(f"{p}: {len(ss)} tiles exceeds limit")
            sizes.append([w, h])
            cams.append(_camera(scene, w, h))
            specs.append([(s.x, s.y, s.width, s.height) for s in ss])
        out = {
            "frame_paths": [str(p) for p in files],
            "tile_specs": specs,
            "image_size": torch.tensor(sizes, dtype=torch.float32),
            "camera_params": torch.stack(cams),
            "poses": poses,
            "scene": scene.name,
        }
        if self.has_gt:
            gp = scene / "panorama.png"
            if not gp.exists():
                raise FileNotFoundError(gp)
            with Image.open(gp) as im:
                out["gt_panorama"] = (
                    torch.from_numpy(np.asarray(im.convert("RGB")))
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
                        torch.from_numpy(np.asarray(crop)).permute(2, 0, 1).float()
                        / 255.0
                    )
            yield fi, torch.stack(tiles), torch.tensor(
                [[x, y] for x, y, _, _ in chosen], dtype=torch.float32
            ), torch.tensor([[w, h] for _, _, w, h in chosen], dtype=torch.float32)


def tile_collate(batch):
    if len(batch) != 1:
        raise ValueError("Native-resolution tiled scenes require batch_size=1")
    return batch[0]
