"""Metadata embeddings for variable-resolution panorama tiles."""
import torch
import torch.nn as nn


class TileMetadataEncoder(nn.Module):
    """Encode tile position, source camera geometry and pose."""
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, dim), nn.GELU(), nn.Linear(dim, dim))

    def forward(self, tile_xy, tile_wh, image_wh, camera_params, rotations):
        rot = rotations.reshape(*rotations.shape[:2], 9)
        geom = torch.cat([tile_xy, tile_wh, image_wh, camera_params[..., :5], rotations.reshape(*rotations.shape[:2], 9)], dim=-1)
        return self.net(geom)
