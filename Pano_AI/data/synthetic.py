import torch
import numpy as np

def sample_poses(num_images):
    yaw = torch.rand(num_images) * 2 * torch.pi - torch.pi
    pitch = torch.zeros(num_images)
    roll = torch.zeros(num_images)
    return torch.stack([yaw, pitch, roll], dim=1)

def generate_synthetic_batch(pano, num_images, fov=90):
    """
    pano: (3, H, W) equirectangular panorama
    returns synthetic images + exact poses
    """
    poses = sample_poses(num_images)
    images = []

    for i in range(num_images):
        # Placeholder: replace with inverse spherical projection
        # For now, simple crop for bootstrapping
        h, w = pano.shape[1:]
        x = int((poses[i,0] + torch.pi) / (2*torch.pi) * w)
        crop = pano[:, :, max(0, x-112):x+112]
        crop = torch.nn.functional.interpolate(
            crop.unsqueeze(0),
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )[0]
        images.append(crop)

    return torch.stack(images), poses
