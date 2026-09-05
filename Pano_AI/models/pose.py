import torch
import torch.nn as nn


def ypr_to_rotation(yaw, pitch, roll):
    """Convert yaw/pitch/roll radians to a camera-to-world rotation matrix."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)
    zero = torch.zeros_like(yaw)
    one = torch.ones_like(yaw)
    rz = torch.stack((torch.stack((cy, -sy, zero)), torch.stack((sy, cy, zero)), torch.stack((zero, zero, one))))
    ry = torch.stack((torch.stack((cp, zero, sp)), torch.stack((zero, one, zero)), torch.stack((-sp, zero, cp))))
    rx = torch.stack((torch.stack((one, zero, zero)), torch.stack((zero, cr, -sr)), torch.stack((zero, sr, cr))))
    return rz @ ry @ rx


def normalize_pose(pose_tensor):
    """[N,3] yaw/pitch/roll degrees -> [N,3,3] rotation matrices."""
    pose_tensor = torch.as_tensor(pose_tensor, dtype=torch.float32)
    pose_rad = pose_tensor * (torch.pi / 180.0)
    if pose_rad.ndim == 2:
        return torch.stack([ypr_to_rotation(y, p, r) for y, p, r in pose_rad])
    if pose_rad.ndim == 3:
        return torch.stack([torch.stack([ypr_to_rotation(y, p, r) for y, p, r in scene]) for scene in pose_rad])
    raise ValueError(f"pose_tensor must be [N,3] or [B,N,3], got {pose_tensor.shape}")


class PoseHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 3)

    def forward(self, x):
        return self.fc(x)
