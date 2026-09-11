import torch
import torch.nn as nn

from models.fusion import SphericalFusion


class PanoramaModel(nn.Module):
    """Shared arbitrary-N panorama model for fisheye and pinhole cameras."""

    def __init__(self, encoder, aggregator, decoder, pano_h, pano_w):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.fusion = SphericalFusion()
        self.decoder = decoder
        self.pano_h = pano_h
        self.pano_w = pano_w
        self.last_spherical = None

    def forward(self, images, rotations, frame_mask=None, camera_params=None):
        if images.ndim != 5:
            raise ValueError(f"images must be [B,N,3,H,W], got {images.shape}")
        b, n = images.shape[:2]
        if frame_mask is None:
            frame_mask = torch.ones(b, n, device=images.device, dtype=torch.bool)
        features = self.encoder(images.reshape(b * n, *images.shape[2:]))
        _, c, fh, fw = features.shape
        features = features.reshape(b, n, c, fh, fw)
        context = self.aggregator(features, frame_mask)
        spherical = self.fusion(features, rotations, self.pano_h, self.pano_w, frame_mask, camera_params)
        self.last_spherical = spherical
        # Broadcast the scene-level context as a learned conditioning signal.
        conditioned = spherical * torch.sigmoid(context).unsqueeze(-1).unsqueeze(-1)
        return self.decoder(conditioned)
