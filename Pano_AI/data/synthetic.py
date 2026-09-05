import torch
import torch.nn.functional as F


def sample_poses(num_images):
    yaw = torch.rand(num_images) * 360.0 - 180.0
    pitch = torch.zeros(num_images)
    roll = torch.zeros(num_images)
    return torch.stack((yaw, pitch, roll), dim=1)  # degrees, [N,3]


def generate_synthetic_batch(pano, num_images, fov=180):
    """Bootstrapping helper using the same [N,3] degree pose contract."""
    _, _, width = pano.shape
    poses = sample_poses(num_images)
    images = []
    for i in range(num_images):
        center = int((poses[i, 0].item() + 180.0) / 360.0 * width) % width
        half = max(1, width // 12)
        idx = torch.arange(center - half, center + half, device=pano.device) % width
        crop = pano[:, :, idx]
        images.append(F.interpolate(crop.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)[0])
    return torch.stack(images), poses
