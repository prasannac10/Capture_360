"""Camera-aware equirectangular projection for native-resolution tile features."""

import math
import torch
import torch.nn.functional as F


def equirect_dirs(h, w, device, dtype):
    lon = torch.linspace(-math.pi, math.pi, w, device=device, dtype=dtype)
    lat = torch.linspace(math.pi / 2, -math.pi / 2, h, device=device, dtype=dtype)
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    return torch.stack(
        (
            torch.cos(lat) * torch.sin(lon),
            torch.sin(lat),
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
    tile_features,
    tile_xy,
    tile_wh,
    image_size,
    camera_params,
    rotations,
    pano_h,
    pano_w,
    feature_stride=8,
):
    b, n, k, c, hf, wf = tile_features.shape
    dev, dtype = tile_features.device, tile_features.dtype
    world = equirect_dirs(pano_h, pano_w, dev, dtype).view(1, 1, pano_h, pano_w, 3)
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
    canvas = torch.zeros(b, c, pano_h, pano_w, device=dev, dtype=dtype)
    weight = torch.zeros(b, 1, pano_h, pano_w, device=dev, dtype=dtype)
    tw = wf * feature_stride
    th = hf * feature_stride
    for i in range(k):
        x0 = tile_xy[:, :, i, 0].view(b, n, 1, 1)
        y0 = tile_xy[:, :, i, 1].view(b, n, 1, 1)
        x1 = x0 + tile_wh[:, :, i, 0].view(b, n, 1, 1)
        y1 = y0 + tile_wh[:, :, i, 1].view(b, n, 1, 1)
        gx = 2 * (u - x0) / (tw - 1) - 1
        gy = 2 * (v - y0) / (th - 1) - 1
        grid = torch.stack((gx, gy), -1).reshape(b * n, pano_h, pano_w, 2)
        fti = tile_features[:, :, i].reshape(b * n, c, hf, wf)
        samp = F.grid_sample(
            fti, grid, align_corners=True, padding_mode="zeros"
        ).reshape(b, n, c, pano_h, pano_w)
        w = ((valid) & (u >= x0) & (u < x1) & (v >= y0) & (v < y1)).to(dtype)
        canvas += (samp * w.unsqueeze(2)).sum(1)
        weight += w.sum(1).unsqueeze(1)
    return canvas / weight.clamp_min(1e-6), weight
