"""High-resolution I/O and tiled inference helpers for Capture360."""
from __future__ import annotations
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
from PIL import Image

OUTPUT_SIZE = (12000, 6000)
SUPPORTED_CAPTURE_PROFILES = {"dslr_fisheye": (9504, 6336), "drone_still": (4096, 3072), "mobile_min": (3000, 4000), "mobile_max": (6120, 8160)}

def validate_capture_size(width: int, height: int, projection: str) -> None:
    w, h = int(width), int(height)
    if w <= 0 or h <= 0: raise ValueError(f"Invalid capture dimensions: {w}x{h}")
    p = projection.lower().replace("-", "_")
    if p in {"fisheye", "fisheye_180", "equidistant"}:
        if (w, h) != SUPPORTED_CAPTURE_PROFILES["dslr_fisheye"]: raise ValueError(f"DSLR fisheye profile requires 9504x6336; got {w}x{h}")
    elif p in {"pinhole", "phone", "drone", "perspective"}:
        short, long = sorted((w, h))
        if short < 3000 or long < 4000 or short > 6120 or long > 8160: raise ValueError(f"Pinhole source {w}x{h} is outside 3000x4000..6120x8160")

def resize_panorama(image: np.ndarray, output_size: tuple[int, int] = OUTPUT_SIZE) -> np.ndarray:
    w, h = output_size
    return image if image.shape[:2] == (h, w) else cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)

def apply_tiled(image: np.ndarray, fn: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray], tile_size: int = 1024, overlap: int = 128, mask: Optional[np.ndarray] = None) -> np.ndarray:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size // 2: raise ValueError("Require tile_size > 0 and 0 <= overlap < tile_size/2")
    h, w = image.shape[:2]; out = np.zeros_like(image, dtype=np.float32); weight = np.zeros((h,w,1), dtype=np.float32); step = tile_size-overlap
    def starts(length):
        values=list(range(0,max(1,length-tile_size+1),step)); last=max(0,length-tile_size)
        if not values or values[-1] != last: values.append(last)
        return values
    for y in starts(h):
        for x in starts(w):
            y2,x2=min(y+tile_size,h),min(x+tile_size,w); tile=image[y:y2,x:x2]; tile_mask=mask[y:y2,x:x2] if mask is not None else None
            corrected=fn(tile,tile_mask)
            if corrected.shape[:2] != tile.shape[:2]: raise ValueError("Correction model changed tile dimensions")
            th,tw=tile.shape[:2]; wy=np.hanning(th) if th>2 else np.ones(th); wx=np.hanning(tw) if tw>2 else np.ones(tw); win=np.maximum(np.outer(wy,wx).astype(np.float32),1e-3)[...,None]
            out[y:y2,x:x2]+=corrected.astype(np.float32)*win; weight[y:y2,x:x2]+=win
    return np.divide(out,np.maximum(weight,1e-6)).clip(0,255).astype(np.uint8)

def save_png_tiff(image: np.ndarray, stem: str | Path) -> None:
    stem=Path(stem); stem.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(image.astype(np.uint8)).save(stem.with_suffix('.png')); Image.fromarray((image.astype(np.uint16)*257) if image.dtype!=np.uint16 else image).save(stem.with_suffix('.tiff'),compression='tiff_deflate')
