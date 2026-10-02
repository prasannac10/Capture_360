"""Focused tests for the variable-resolution tiled data contract."""

from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from panorama.pano_ai.data.tiling import tile_specs
from panorama.pano_ai.data.tile_dataset import VariableTilePanoramaDataset
from panorama.pano_ai.models.encoder import ImageEncoder


def main():
    cases = [(6336, 9504), (3072, 4096), (4000, 3000), (8160, 6120)]
    for h, w in cases:
        s = tile_specs(h, w, 1024, 128)
        assert len(s) > 0
        assert min(x.width for x in s) > 0 and min(x.height for x in s) > 0
        assert all(x.x + x.width <= w and x.y + x.height <= h for x in s)
    e = ImageEncoder(16, pretrained=False)
    assert e.feature_stride == 8
    y = e(torch.zeros(1, 3, 1024, 1024))
    assert y.shape == (1, 16, 128, 128), y.shape
    print(
        "VARIABLE-TILED SMOKE PASS: tile geometry, native resolutions, encoder stride-8"
    )


if __name__ == "__main__":
    main()
