import torch
import torch.nn as nn

def spherical_project(features, poses):
    """
    features: (B, N, D)
    poses:    (B, N, 2)  yaw, pitch in degrees
    """
    yaw = poses[..., 0] / 360.0
    pitch = poses[..., 1] / 180.0

    weight = 1.0 + yaw.unsqueeze(-1) + pitch.unsqueeze(-1)
    return features * weight

class SphericalFusion(nn.Module):
    def forward(self, feats, poses):
        return spherical_project(feats, poses)
