import torch
from models.spherical import pano_to_views


def _smoothness_loss(pano):
    dx = (pano[..., :, 1:] - pano[..., :, :-1]).abs().mean()
    dy = (pano[..., 1:, :] - pano[..., :-1, :]).abs().mean()
    return dx + dy


def geometry_loss(pano, images=None, rotations=None, frame_mask=None, smoothness_weight=1.0):
    """Photometric multi-view reprojection consistency plus panorama smoothness."""
    if images is None or rotations is None:
        raise ValueError("geometry_loss requires images and rotations for geometry-only training")
    views, valid = pano_to_views(pano, rotations, images.shape[-2], images.shape[-1], frame_mask)
    valid = valid.to(images.dtype).view(1, 1, 1, *valid.shape)
    photometric_mask = valid.expand_as(views)
    if frame_mask is not None:
        photometric_mask = photometric_mask * frame_mask.to(images.dtype).view(images.shape[0], images.shape[1], 1, 1, 1)
    photometric = (views - images).abs() * photometric_mask
    denom = photometric_mask.sum().clamp_min(1.0)
    return photometric.sum() / denom + smoothness_weight * _smoothness_loss(pano)
