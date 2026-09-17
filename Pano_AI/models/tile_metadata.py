"""Normalized tile/camera/pose metadata encoder."""
import torch
import torch.nn as nn
class TileMetadataEncoder(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.net=nn.Sequential(nn.Linear(21,dim),nn.GELU(),nn.Linear(dim,dim))
    def forward(self,tile_xy,tile_wh,image_wh,camera_params,rotations):
        geom=torch.cat([tile_xy,tile_wh,image_wh,camera_params,rotations.reshape(*rotations.shape[:-2],9)],dim=-1)
        return self.net(geom)
