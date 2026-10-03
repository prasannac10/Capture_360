"""Camera-aware equirectangular projection for native-resolution tile features."""

import math
import torch
import torch.nn.functional as F

PANORAMA_CONTRACT = 'panorama_exposure_blending_v4'


def panorama_contract(detail_mode='features'):
    if detail_mode == 'features':
        return PANORAMA_CONTRACT
    if detail_mode == 'rgb_residual':
        return 'panorama_native_rgb_residual_v6'
    raise ValueError('Unknown panorama detail mode')


def normalize_camera_features(canvas, tile_weight, camera_weight):
    """Normalize tiles before applying the camera's footprint weight."""
    weight = camera_weight * (tile_weight > 0).to(camera_weight.dtype)
    return canvas / tile_weight.clamp_min(1e-12), weight


def _cosine_edge_weight(distance, width):
    phase = (distance / width).clamp(0, 1)
    return (0.5 - 0.5 * torch.cos(math.pi * phase)).clamp_min(1e-3)


def equirect_dirs(h, w, device, dtype):
    lon = ((torch.arange(w, device=device, dtype=dtype) + .5) / w - .5) * (2 * math.pi)
    lat = (.5 - (torch.arange(h, device=device, dtype=dtype) + .5) / h) * math.pi
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    return torch.stack(
        (
            torch.cos(lat) * torch.sin(lon),
            -torch.sin(lat),
            torch.cos(lat) * torch.cos(lon),
        ),
        -1,
    )


def ypr_to_rot(ypr):
    y, p, r = (
        torch.deg2rad(ypr[:, 0]),
        torch.deg2rad(ypr[:, 1]),
        torch.deg2rad(ypr[:, 2]),
    )
    z = torch.zeros_like(y)
    o = torch.ones_like(y)
    cy, sy = torch.cos(y), torch.sin(y)
    cp, sp = torch.cos(p), torch.sin(p)
    cr, sr = torch.cos(r), torch.sin(r)
    Ry = torch.stack((cy, z, sy, z, o, z, -sy, z, cy), -1).reshape(-1, 3, 3)
    Rx = torch.stack((o, z, z, z, cp, -sp, z, sp, cp), -1).reshape(-1, 3, 3)
    Rz = torch.stack((cr, -sr, z, sr, cr, z, z, z, o), -1).reshape(-1, 3, 3)
    return Ry @ Rx @ Rz


def project_tile_features(
    tile_features, tile_xy, tile_wh, image_size, camera_params, rotations,
    pano_h, pano_w, feature_stride=8, return_per_camera=False,
):
    """Fuse tiles within cameras, then feather camera footprints.

    Streaming callers request per-camera weighted sums, tile weights and camera
    weights, accumulate all chunks of a camera, and only then normalize it.
    """
    # Half precision cannot represent all native pixel coordinates above 2048.
    with torch.autocast(device_type=tile_features.device.type, enabled=False):
        return _project_tile_features(
            tile_features.float(), tile_xy.float(), tile_wh.float(), image_size.float(),
            camera_params.float(), rotations.float(), pano_h, pano_w, feature_stride,
            return_per_camera)


def project_camera_pixels(image_size, camera_params, rotations, pano_h, pano_w, region=None):
    """Shared camera projection for feature fusion and overlap photometry (float32)."""
    b, n = image_size.shape[:2]
    dev, dtype = image_size.device, image_size.dtype
    if region is None:
        world = equirect_dirs(pano_h, pano_w, dev, dtype)
    else:
        top, left, height, width = region
        lon = ((torch.arange(left, left + width, device=dev, dtype=dtype) + .5) / pano_w - .5) * (2 * math.pi)
        rows = torch.arange(top, top + height, device=dev, dtype=dtype).clamp(0, pano_h - 1)
        lat = (.5 - (rows + .5) / pano_h) * math.pi
        lat, lon = torch.meshgrid(lat, lon, indexing='ij')
        world = torch.stack((torch.cos(lat) * torch.sin(lon), -torch.sin(lat),
                             torch.cos(lat) * torch.cos(lon)), -1)
    rh, rw = world.shape[:2]
    world = world.view(1, 1, rh, rw, 3)
    dirs = torch.einsum(
        "bnij,bnhwj->bnhwi", rotations.transpose(-1, -2), world.expand(b, n, -1, -1, -1)
    )
    z = dirs[..., 2].clamp(-1, 1)
    theta = torch.acos(z)
    rho = torch.sqrt(dirs[..., 0] ** 2 + dirs[..., 1] ** 2).clamp_min(1e-8)
    fov = torch.deg2rad(camera_params[..., 5]).view(b, n, 1, 1).clamp_min(1e-4)
    fx, fy, cx, cy = [camera_params[..., i].view(b, n, 1, 1) for i in range(4)]
    pin = camera_params[..., 4].view(b, n, 1, 1) > 0.5
    uf = fx * theta * dirs[..., 0] / rho + cx
    vf = fy * theta * dirs[..., 1] / rho + cy
    zsafe = dirs[..., 2].clamp_min(1e-6)
    up = fx * dirs[..., 0] / zsafe + cx
    vp = fy * dirs[..., 1] / zsafe + cy
    u = torch.where(pin, up, uf)
    v = torch.where(pin, vp, vf)
    valid = torch.where(
        pin,
        (dirs[..., 2] > 0)
        & (up >= 0)
        & (up < image_size[..., 0].view(b, n, 1, 1))
        & (vp >= 0)
        & (vp < image_size[..., 1].view(b, n, 1, 1)),
        (theta <= fov / 2)
        & (uf >= 0)
        & (uf < image_size[..., 0].view(b, n, 1, 1))
        & (vf >= 0)
        & (vf < image_size[..., 1].view(b, n, 1, 1)),
    )
    iw = image_size[..., 0].view(b, n, 1, 1)
    ih = image_size[..., 1].view(b, n, 1, 1)
    edge = torch.minimum(torch.minimum(u / iw, (iw - u) / iw),
                         torch.minimum(v / ih, (ih - v) / ih))
    # Feather the outer 10% of each image; fisheye coverage also has a circular edge.
    edge = torch.where(pin, edge, torch.minimum(edge, (fov / 2 - theta) / fov))
    camera_weight = (_cosine_edge_weight(edge, .1) * valid).unsqueeze(2)
    return u, v, valid, camera_weight


def _project_tile_features(
    tile_features,
    tile_xy,
    tile_wh,
    image_size,
    camera_params,
    rotations,
    pano_h,
    pano_w,
    feature_stride=8,
    return_per_camera=False,
):
    b, n, k, c, hf, wf = tile_features.shape
    dev, dtype = tile_features.device, tile_features.dtype
    u, v, valid, camera_weight = project_camera_pixels(
        image_size, camera_params, rotations, pano_h, pano_w)
    canvas = torch.zeros(b, n, c, pano_h, pano_w, device=dev, dtype=dtype)
    weight = torch.zeros(b, n, 1, pano_h, pano_w, device=dev, dtype=dtype)
    for i in range(k):
        x0 = tile_xy[:, :, i, 0].view(b, n, 1, 1)
        y0 = tile_xy[:, :, i, 1].view(b, n, 1, 1)
        x1 = x0 + tile_wh[:, :, i, 0].view(b, n, 1, 1)
        y1 = y0 + tile_wh[:, :, i, 1].view(b, n, 1, 1)
        # Stride-8 ResNet features are centred at source pixels 0,8,16,... .
        gx = 2 * (u - x0) / max(feature_stride * (wf - 1), 1) - 1
        gy = 2 * (v - y0) / max(feature_stride * (hf - 1), 1) - 1
        grid = torch.stack((gx, gy), -1).reshape(b * n, pano_h, pano_w, 2)
        fti = tile_features[:, :, i].reshape(b * n, c, hf, wf)
        samp = F.grid_sample(
            fti, grid, align_corners=True, padding_mode="border"
        ).reshape(b, n, c, pano_h, pano_w)
        w = ((valid) & (u >= x0) & (u < x1) & (v >= y0) & (v < y1)).to(dtype)
        # Continuous raised-cosine tile weights suppress encoder boundary effects.
        tx = ((u - x0) / (x1 - x0).clamp_min(1)).clamp(0, 1)
        ty = ((v - y0) / (y1 - y0).clamp_min(1)).clamp(0, 1)
        w = w * torch.sin(math.pi * tx).square().clamp_min(1e-3)
        w = w * torch.sin(math.pi * ty).square().clamp_min(1e-3)
        canvas = canvas + samp * w.unsqueeze(2)
        weight = weight + w.unsqueeze(2)
    if return_per_camera:
        return canvas, weight, camera_weight
    features, camera_weight = normalize_camera_features(canvas, weight, camera_weight)
    total = camera_weight.sum(1)
    return (features * camera_weight).sum(1) / total.clamp_min(1e-12), total
