"""Architecture smoke tests; no customer dataset is required."""

import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.aggregator import SetAggregator
from models.color_enhance import ColorEnhancementUNet
from models.decoder import PanoramaDecoder
from models.encoder import ImageEncoder
from models.glare_removal import GlareRemovalUNet
from models.nadir_zenith import NadirZenithInpainter
from models.panorama_model import PanoramaModel


def _batch(batch_size, frames):
    images = torch.rand(batch_size, frames, 3, 224, 224)
    rotations = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch_size, frames, 1, 1)
    mask = torch.ones(batch_size, frames, dtype=torch.bool)
    camera = torch.zeros(batch_size, frames, 5)
    camera[..., 0] = 1.0
    camera[..., 1] = 1.0
    return images, rotations, mask, camera


def _model(cfg):
    dim = cfg["model"]["feature_dim"]
    return PanoramaModel(ImageEncoder(dim, pretrained=False), SetAggregator(dim), PanoramaDecoder(dim), 256, 512)


def main():
    with open(ROOT / "config.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    model = _model(cfg)
    for frames in (6, 24):
        images, rotations, mask, camera = _batch(1, frames)
        if frames == 24:
            camera[..., 4] = 1.0
        pred = model(images, rotations, mask, camera)
        assert tuple(pred.shape) == (1, 3, 256, 512), pred.shape
        loss = pred.mean()
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        model.zero_grad(set_to_none=True)

    # Mixed-N padded batch: fisheye scene has six valid frames, phone scene 24.
    images = torch.rand(2, 24, 3, 224, 224)
    rotations = torch.eye(3).reshape(1, 1, 3, 3).repeat(2, 24, 1, 1)
    mask = torch.zeros(2, 24, dtype=torch.bool)
    mask[0, :6] = True
    mask[1, :24] = True
    camera = torch.zeros(2, 24, 5)
    camera[..., :2] = 1.0
    camera[1, :, 4] = 1.0
    pred = model(images, rotations, mask, camera)
    assert tuple(pred.shape) == (2, 3, 256, 512)

    x = torch.rand(2, 3, 128, 256)
    for stage in (GlareRemovalUNet(), ColorEnhancementUNet()):
        y = stage(x)
        assert y.shape == x.shape and torch.isfinite(y).all()
        y.mean().backward()
    pole = NadirZenithInpainter()
    y = pole(x, torch.ones(2, 1, 128, 256))
    assert y.shape == x.shape and torch.isfinite(y).all()
    print("SMOKE PASS: fisheye N=6, phone N=24, mixed-N batch, panorama decoder, and correction U-Nets")


if __name__ == "__main__":
    main()
