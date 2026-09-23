"""Original-resolution overlapping tiling utilities for Capture360."""

from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TileSpec:
    index: int
    x: int
    y: int
    width: int
    height: int
    image_width: int
    image_height: int

    @property
    def normalized_xy(self) -> Tuple[float, float]:
        return (
            self.x / max(1, self.image_width - 1),
            self.y / max(1, self.image_height - 1),
        )


def tile_specs(
    height: int, width: int, tile_size: int = 1024, overlap: int = 128
) -> List[TileSpec]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size must be > 0 and 0 <= overlap < tile_size")
    stride = tile_size - overlap
    ys = list(range(0, max(1, height - tile_size + 1), stride))
    xs = list(range(0, max(1, width - tile_size + 1), stride))
    if not ys or ys[-1] + tile_size < height:
        ys.append(max(0, height - tile_size))
    if not xs or xs[-1] + tile_size < width:
        xs.append(max(0, width - tile_size))
    specs = []
    idx = 0
    for y in ys:
        for x in xs:
            specs.append(
                TileSpec(
                    idx,
                    x,
                    y,
                    min(tile_size, width - x),
                    min(tile_size, height - y),
                    width,
                    height,
                )
            )
            idx += 1
    return specs


def extract_tiles(image: torch.Tensor, tile_size: int = 1024, overlap: int = 128):
    """Extract padded [N,3,tile,tile] tiles and their geometry from [3,H,W]."""
    if image.ndim != 3:
        raise ValueError(f"image must be [C,H,W], got {tuple(image.shape)}")
    _, h, w = image.shape
    specs = tile_specs(h, w, tile_size, overlap)
    tiles = []
    for s in specs:
        crop = image[:, s.y : s.y + s.height, s.x : s.x + s.width]
        pad_h, pad_w = tile_size - crop.shape[-2], tile_size - crop.shape[-1]
        if pad_h or pad_w:
            crop = F.pad(
                crop,
                (0, pad_w, 0, pad_h),
                mode="reflect" if min(crop.shape[-2:]) > 1 else "replicate",
            )
        tiles.append(crop)
    return torch.stack(tiles), specs


def raised_cosine_window(tile_size: int, device, dtype):
    # Hann-like 2-D weighting reduces visible seams when tiled features are fused.
    w = torch.hann_window(
        tile_size, periodic=False, device=device, dtype=dtype
    ).clamp_min(1e-3)
    return torch.outer(w, w)
