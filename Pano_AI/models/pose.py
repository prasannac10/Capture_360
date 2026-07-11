# pose.py
import torch
import math

def ypr_to_rotation(yaw, pitch, roll):
    """
    yaw, pitch, roll: radians
    returns: [3,3] rotation matrix
    """
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    Rz = torch.tensor([
        [cy, -sy, 0],
        [sy,  cy, 0],
        [ 0,   0, 1]
    ])

    Ry = torch.tensor([
        [ cp, 0, sp],
        [  0, 1,  0],
        [-sp, 0, cp]
    ])

    Rx = torch.tensor([
        [1,  0,   0],
        [0, cr, -sr],
        [0, sr,  cr]
    ])

    return Rz @ Ry @ Rx


def normalize_pose(pose_tensor):
    """
    pose_tensor: [N,3] yaw, pitch, roll in degrees
    returns: [N,3,3] rotation matrices
    """
    pose_rad = pose_tensor * math.pi / 180.0
    rotations = []

    for y, p, r in pose_rad:
        R = ypr_to_rotation(y, p, r)
        rotations.append(R)

    return torch.stack(rotations)

class PoseHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 3)  # yaw, pitch, roll

    def forward(self, x):
        return self.fc(x)  # (N, 3)
