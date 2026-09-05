import math
import torch
import torch.nn.functional as F


def _equirect_dirs(height, width, device, dtype):
    theta = torch.linspace(-math.pi, math.pi, width, device=device, dtype=dtype)
    phi = torch.linspace(-math.pi / 2, math.pi / 2, height, device=device, dtype=dtype)
    phi, theta = torch.meshgrid(phi, theta, indexing="ij")
    return torch.stack((torch.cos(phi) * torch.sin(theta), torch.sin(phi), torch.cos(phi) * torch.cos(theta)), dim=-1)


def _fisheye_grid(camera_dirs):
    """Project unit camera rays through an assumed 180-degree equidistant fisheye."""
    z = camera_dirs[..., 2].clamp(-1.0, 1.0)
    theta = torch.acos(z)
    xy_norm = torch.sqrt(camera_dirs[..., 0] ** 2 + camera_dirs[..., 1] ** 2).clamp_min(1e-8)
    radius = theta / (math.pi / 2.0)
    u = radius * camera_dirs[..., 0] / xy_norm
    v = radius * camera_dirs[..., 1] / xy_norm
    valid = (theta <= math.pi / 2.0) & (radius <= 1.0)
    return torch.stack((u, v), dim=-1), valid


def spherical_project(features, rotations, pano_h, pano_w, frame_mask=None):
    """Differentiable inverse warp from [B,N,C,h,w] views into [B,C,pano_h,pano_w]."""
    if features.ndim != 5:
        raise ValueError(f"features must be [B,N,C,h,w], got {features.shape}")
    bsz, num_views, channels, _, _ = features.shape
    device, dtype = features.device, features.dtype
    world_dirs = _equirect_dirs(pano_h, pano_w, device, dtype).unsqueeze(0).expand(bsz, -1, -1, -1)
    canvas = torch.zeros(bsz, channels, pano_h, pano_w, device=device, dtype=dtype)
    weights = torch.zeros(bsz, 1, pano_h, pano_w, device=device, dtype=dtype)
    for i in range(num_views):
        camera_dirs = torch.einsum("bhwc,bcj->bhwj", world_dirs, rotations[:, i])
        grid, valid = _fisheye_grid(camera_dirs)
        sampled = F.grid_sample(features[:, i], grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        visibility = valid.to(dtype) * camera_dirs[..., 2].clamp_min(0.0)
        if frame_mask is not None:
            visibility = visibility * frame_mask[:, i].to(dtype).view(bsz, 1, 1)
        canvas = canvas + sampled * visibility.unsqueeze(1)
        weights = weights + visibility.unsqueeze(1)
    return canvas / weights.clamp_min(1e-6)


def pano_to_views(pano, rotations, image_h, image_w, frame_mask=None):
    """Differentiably reproject an equirectangular panorama into fisheye views."""
    bsz, _, _, _ = pano.shape
    device, dtype = pano.device, pano.dtype
    yy, xx = torch.meshgrid(torch.linspace(-1.0, 1.0, image_h, device=device, dtype=dtype), torch.linspace(-1.0, 1.0, image_w, device=device, dtype=dtype), indexing="ij")
    radius = torch.sqrt(xx * xx + yy * yy)
    theta = radius.clamp_max(1.0) * (math.pi / 2.0)
    sin_theta = torch.sin(theta)
    xy_norm = radius.clamp_min(1e-8)
    camera_dirs = torch.stack((sin_theta * xx / xy_norm, sin_theta * yy / xy_norm, torch.cos(theta)), dim=-1)
    valid = radius <= 1.0
    outputs = []
    for i in range(rotations.shape[1]):
        world_dirs = torch.einsum("hwc,bcj->bhwj", camera_dirs, rotations[:, i].transpose(1, 2))
        x = torch.atan2(world_dirs[..., 0], world_dirs[..., 2]) / math.pi
        y = world_dirs[..., 1] / (math.pi / 2.0)
        sampled = F.grid_sample(pano, torch.stack((x, y), dim=-1), mode="bilinear", padding_mode="border", align_corners=True)
        if frame_mask is not None:
            sampled = sampled * frame_mask[:, i].to(dtype).view(bsz, 1, 1, 1)
        outputs.append(sampled)
    return torch.stack(outputs, dim=1), valid
