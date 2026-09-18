"""Non-ML fisheye-to-equirectangular baseline using OpenCV."""
from pathlib import Path
import cv2
import numpy as np


def fisheye_to_equirectangular(image, output_size=(512, 256), fov_degrees=180.0):
    w, h = output_size
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    lon = (xx / w - 0.5) * 2.0 * np.pi
    lat = (0.5 - yy / h) * np.pi
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    theta = np.arctan2(x, z)
    phi = np.arctan2(y, np.sqrt(x*x + z*z))
    r = np.tan(np.deg2rad(fov_degrees) / 2.0)
    scale = np.sqrt(theta*theta + phi*phi) / (np.pi / 2.0)
    radius = scale * (min(image.shape[:2]) / 2.0)
    src_x = image.shape[1] / 2.0 + radius * np.sin(theta) / np.maximum(scale, 1e-6)
    src_y = image.shape[0] / 2.0 - radius * np.sin(phi) / np.maximum(scale, 1e-6)
    valid = scale <= 1.0
    src_x = np.mod(src_x, image.shape[1]).astype(np.float32)
    src_y = np.clip(src_y, 0, image.shape[0]-1).astype(np.float32)
    out = cv2.remap(image, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    out[~valid] = 0
    return out


def stitch_fisheye_files(image_paths, output_path, output_size=(512, 256), fov_degrees=180.0):
    frames = [cv2.imread(str(p)) for p in image_paths]
    if not frames or any(x is None for x in frames):
        raise ValueError('Could not read one or more fisheye images')
    projected = [fisheye_to_equirectangular(x, output_size, fov_degrees) for x in frames]
    valid = [np.any(x != 0, axis=2) for x in projected]
    acc = np.zeros_like(projected[0], dtype=np.float32); weight = np.zeros(projected[0].shape[:2], dtype=np.float32)
    for frame, mask in zip(projected, valid):
        acc += frame.astype(np.float32) * mask[..., None]
        weight += mask.astype(np.float32)
    pano = (acc / np.maximum(weight[..., None], 1.0)).clip(0,255).astype(np.uint8)
    cv2.imwrite(str(output_path), pano)
    return pano


def stitch_perspective_files(image_paths, output_path, output_size):
    """Feature-based OpenCV stitcher for drone and phone pinhole frames."""
    frames = [cv2.imread(str(p)) for p in image_paths]
    if len(frames) < 2 or any(frame is None for frame in frames):
        raise ValueError("Could not read at least two perspective images")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA) if hasattr(cv2, "Stitcher_create") else cv2.Stitcher.create(cv2.Stitcher.PANORAMA)
    status, panorama = stitcher.stitch(frames)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"OpenCV perspective stitching failed (status={status})")
    panorama = cv2.resize(panorama, output_size, interpolation=cv2.INTER_LANCZOS4)
    if not cv2.imwrite(str(output_path), panorama):
        raise OSError(f"Could not write panorama: {output_path}")
    return panorama


def stitch_files(image_paths, output_path, output_size, profile):
    """Select fisheye projection or perspective stitching from a capture profile."""
    if profile["projection"] == "fisheye_180":
        return stitch_fisheye_files(image_paths, output_path, output_size, profile.get("fov_degrees", 180.0))
    if profile["projection"] == "pinhole":
        return stitch_perspective_files(image_paths, output_path, output_size)
    raise ValueError(f"Unsupported classical projection: {profile['projection']}")

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('images', nargs='+'); p.add_argument('output'); a=p.parse_args()
    stitch_fisheye_files(a.images, a.output)
