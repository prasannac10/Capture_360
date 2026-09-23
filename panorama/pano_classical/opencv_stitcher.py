"""Non-ML fisheye-to-equirectangular baseline using OpenCV."""

from pathlib import Path
import logging
import cv2
import numpy as np
from .raw_pipeline import stitch_raw_files

LOGGER = logging.getLogger(__name__)


def fisheye_to_equirectangular(image, output_size=(512, 256), fov_degrees=180.0):
    w, h = output_size
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    lon = (xx / w - 0.5) * 2.0 * np.pi
    lat = (0.5 - yy / h) * np.pi
    x = np.cos(lat) * np.sin(lon)
    y = np.sin(lat)
    z = np.cos(lat) * np.cos(lon)
    theta = np.arctan2(x, z)
    phi = np.arctan2(y, np.sqrt(x * x + z * z))
    r = np.tan(np.deg2rad(fov_degrees) / 2.0)
    scale = np.sqrt(theta * theta + phi * phi) / (np.pi / 2.0)
    radius = scale * (min(image.shape[:2]) / 2.0)
    src_x = image.shape[1] / 2.0 + radius * np.sin(theta) / np.maximum(scale, 1e-6)
    src_y = image.shape[0] / 2.0 - radius * np.sin(phi) / np.maximum(scale, 1e-6)
    valid = scale <= 1.0
    src_x = np.mod(src_x, image.shape[1]).astype(np.float32)
    src_y = np.clip(src_y, 0, image.shape[0] - 1).astype(np.float32)
    out = cv2.remap(
        image, src_x, src_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    out[~valid] = 0
    return out


def stitch_fisheye_files(
    image_paths, output_path, output_size=(512, 256), fov_degrees=180.0
):
    """Stream high-resolution fisheye files using a bounded work canvas."""
    paths = list(image_paths)
    if not paths:
        raise ValueError("Provide one or more fisheye images")
    # Full 12k x 6k float accumulation is ~864 MB before masks and remap
    # buffers.  Compose at 4096 x 2048, then upscale only the final uint8 data.
    work_width = min(int(output_size[0]), 4096)
    work_height = min(int(output_size[1]), max(1, work_width // 2))
    acc = np.zeros((work_height, work_width, 3), dtype=np.float32)
    weight = np.zeros((work_height, work_width), dtype=np.float32)
    for path in paths:
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not read fisheye image: {path}")
        source_scale = min(1.0, 4096 / max(frame.shape[:2]))
        if source_scale < 1.0:
            frame = cv2.resize(
                frame, None, fx=source_scale, fy=source_scale, interpolation=cv2.INTER_AREA
            )
        projected = fisheye_to_equirectangular(
            frame, (work_width, work_height), fov_degrees
        )
        mask = np.any(projected != 0, axis=2)
        acc += projected.astype(np.float32) * mask[..., None]
        weight += mask.astype(np.float32)
    pano = (acc / np.maximum(weight[..., None], 1.0)).clip(0, 255).astype(np.uint8)
    if tuple(output_size) != (work_width, work_height):
        pano = cv2.resize(pano, output_size, interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(str(output_path), pano)
    return pano


def stitch_perspective_files(image_paths, output_path, output_size):
    """Feature-based OpenCV stitcher for drone and phone pinhole frames."""
    frames = [cv2.imread(str(p)) for p in image_paths]
    if len(frames) < 2 or any(frame is None for frame in frames):
        raise ValueError("Could not read at least two perspective images")
    # OpenCV's detail matcher runs a two-neighbour KNN search.  A blank,
    # heavily blurred, or otherwise featureless frame makes that native code
    # assert instead of returning a normal stitch status.  Filter those frames
    # before handing the set to Stitcher.
    detector = cv2.ORB_create(nfeatures=1000)
    usable = []
    skipped = []
    for path, frame in zip(image_paths, frames):
        keypoints = detector.detect(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
        if len(keypoints) >= 8:
            usable.append(frame)
        else:
            skipped.append(str(path))
    if skipped:
        LOGGER.warning(
            "Skipping %d frame(s) with fewer than 8 ORB features: %s",
            len(skipped),
            ", ".join(skipped),
        )
    if len(usable) < 2:
        raise ValueError(
            "Need at least two feature-rich images to stitch; check for blank, "
            "blurred, or corrupt source frames."
        )
    # A phone sweep commonly contains many near-duplicate frames.  Pairwise
    # matching scales quadratically, so retain evenly spaced coverage rather
    # than making a 42+ image set impractically slow.
    max_stitch_frames = 24
    if len(usable) > max_stitch_frames:
        indices = np.linspace(0, len(usable) - 1, max_stitch_frames).round().astype(int)
        usable = [usable[index] for index in indices]
        LOGGER.warning(
            "Using %d evenly spaced frames from %d usable inputs for mobile stitching",
            len(usable),
            len(frames) - len(skipped),
        )
    stitcher = (
        cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        if hasattr(cv2, "Stitcher_create")
        else cv2.Stitcher.create(cv2.Stitcher.PANORAMA)
    )
    # Registration and seam estimation do not benefit from full-HD input.
    # Final output is resized after OpenCV's compositing stage below.
    stitcher.setRegistrationResol(0.3)
    stitcher.setSeamEstimationResol(0.1)
    stitcher.setCompositingResol(6.0)
    status, panorama = stitcher.stitch(usable)
    if status != cv2.Stitcher_OK:
        raise RuntimeError(f"OpenCV perspective stitching failed (status={status})")
    panorama = cv2.resize(panorama, output_size, interpolation=cv2.INTER_LANCZOS4)
    if not cv2.imwrite(str(output_path), panorama):
        raise OSError(f"Could not write panorama: {output_path}")
    return panorama


def stitch_files(image_paths, output_path, output_size, profile):
    """Run the appropriate complete classical OpenCV panorama pipeline.

    For phone/pinhole captures, OpenCV's panorama implementation is used
    because it estimates all cameras globally and bundle-adjusts them before
    warping, seam finding, and multi-band blending.  Chaining independent
    pairwise homographies is not geometrically stable for a camera sweep.
    The explicit raw pipeline remains available for controlled planar inputs.
    """
    if profile["projection"] == "pinhole":
        return stitch_perspective_files(image_paths, output_path, output_size)
    if profile["projection"] == "fisheye_180":
        return stitch_fisheye_files(
            image_paths, output_path, output_size, profile.get("fov_degrees", 180.0)
        )
    raise ValueError(f"Unsupported classical projection: {profile['projection']}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("output")
    a = p.parse_args()
    stitch_fisheye_files(a.images, a.output)
