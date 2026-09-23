"""Explicit feature-based classical panorama pipeline.

The module intentionally keeps the geometric stages visible: source frames are
matched with ORB, chained with RANSAC homographies, warped onto one canvas,
and composited with seam-aware feather weights.  It is the non-ML, raw output
route used before optional correction stages.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


class RawClassicalPanorama:
    """Build a panorama from ordinary OpenCV-readable image files."""

    def __init__(
        self, max_features: int = 2500, ratio_test: float = 0.75, work_side: int = 1280
    ):
        # SIFT is considerably more reliable than ORB for phone rotations and
        # exposure shifts.  Use ORB only on OpenCV builds without SIFT.
        self.detector = (
            cv2.SIFT_create(nfeatures=max_features)
            if hasattr(cv2, "SIFT_create")
            else cv2.ORB_create(nfeatures=max_features)
        )
        self.matcher = cv2.BFMatcher(
            cv2.NORM_L2 if hasattr(cv2, "SIFT_create") else cv2.NORM_HAMMING
        )
        self.ratio_test = ratio_test
        self.work_side = work_side

    def _features(
        self, image: np.ndarray
    ) -> tuple[list[cv2.KeyPoint], np.ndarray | None, float, float]:
        """Extract once at a bounded resolution, retaining source coordinates."""
        height, width = image.shape[:2]
        scale = min(1.0, self.work_side / max(height, width))
        working = (
            cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if scale < 1.0
            else image
        )
        points, descriptors = self.detector.detectAndCompute(
            cv2.cvtColor(working, cv2.COLOR_BGR2GRAY), None
        )
        return points, descriptors, 1.0 / scale, 1.0 / scale

    def _homography(
        self,
        left: tuple[list[cv2.KeyPoint], np.ndarray | None, float, float],
        right: tuple[list[cv2.KeyPoint], np.ndarray | None, float, float],
    ) -> np.ndarray:
        """Return the transform that maps ``right`` into ``left`` coordinates."""
        points_l, desc_l, scale_lx, scale_ly = left
        points_r, desc_r, scale_rx, scale_ry = right
        if desc_l is None or desc_r is None:
            raise RuntimeError("Feature matching failed: a frame has no descriptors")
        pairs = self.matcher.knnMatch(desc_r, desc_l, k=2)
        good = [a for a, b in pairs if a.distance < self.ratio_test * b.distance]
        if len(good) < 4:
            raise RuntimeError(f"Feature matching found only {len(good)} usable matches")
        source = np.float32(
            [(points_r[m.queryIdx].pt[0] * scale_rx, points_r[m.queryIdx].pt[1] * scale_ry) for m in good]
        )
        target = np.float32(
            [(points_l[m.trainIdx].pt[0] * scale_lx, points_l[m.trainIdx].pt[1] * scale_ly) for m in good]
        )
        homography, inliers = cv2.findHomography(source, target, cv2.RANSAC, 4.0)
        if homography is None or inliers is None or int(inliers.sum()) < 8:
            raise RuntimeError("Homography estimation failed")
        return homography

    def _best_connection(
        self,
        frames: list[np.ndarray],
        features: list[tuple[list[cv2.KeyPoint], np.ndarray | None, float, float]],
        connected: dict[int, np.ndarray],
        candidate: int,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Find the strongest usable link from a frame to the current mosaic."""
        best: tuple[np.ndarray, np.ndarray] | None = None
        for parent in sorted(connected, key=lambda index: abs(index - candidate)):
            parent_transform = connected[parent]
            try:
                homography = self._homography(features[parent], features[candidate])
                # Re-evaluate inlier count for ranking.  A strong link matters
                # more than a filename/glob ordering assumption.
                height, width = frames[candidate].shape[:2]
                src = np.float32([[[0, 0], [width, 0], [width, height], [0, height]]])
                projected = cv2.perspectiveTransform(src, homography)
                if not np.isfinite(projected).all():
                    continue
                span = projected[0].max(axis=0) - projected[0].min(axis=0)
                parent_height, parent_width = frames[parent].shape[:2]
                # A real adjacent camera view should retain a plausible image
                # footprint.  This filters degenerate RANSAC solutions that
                # otherwise create enormous, unusable panorama canvases.
                if (
                    span[0] < width * 0.08
                    or span[1] < height * 0.08
                    or span[0] > max(width, parent_width) * 5
                    or span[1] > max(height, parent_height) * 5
                ):
                    continue
                if best is None:
                    best = (parent_transform @ homography, homography)
                    # In normal capture order, this nearest successful frame
                    # avoids expensive comparisons against every older frame.
                    break
            except RuntimeError:
                continue
        return best

    @staticmethod
    def _canvas(transforms: list[np.ndarray], frames: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, int]]:
        corners = []
        for transform, frame in zip(transforms, frames):
            h, w = frame.shape[:2]
            corners.append(cv2.perspectiveTransform(
                np.float32([[[0, 0], [w, 0], [w, h], [0, h]]]), transform
            )[0])
        all_corners = np.concatenate(corners)
        xmin, ymin = np.floor(all_corners.min(axis=0)).astype(int)
        xmax, ymax = np.ceil(all_corners.max(axis=0)).astype(int)
        translation = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], dtype=np.float64)
        return translation, (max(1, xmax - xmin), max(1, ymax - ymin))

    @staticmethod
    def _seam_weight(existing: np.ndarray, incoming: np.ndarray, overlap: np.ndarray) -> np.ndarray:
        """Estimate a low-difference seam and return a smooth incoming weight."""
        difference = cv2.absdiff(existing, incoming).mean(axis=2).astype(np.float32)
        # A blurred difference map makes the seam avoid broad disagreement,
        # while distance from each frame boundary makes transitions gradual.
        difference = cv2.GaussianBlur(difference, (0, 0), 3)
        old_distance = cv2.distanceTransform((np.any(existing != 0, axis=2)).astype(np.uint8), cv2.DIST_L2, 3)
        new_distance = cv2.distanceTransform((np.any(incoming != 0, axis=2)).astype(np.uint8), cv2.DIST_L2, 3)
        weight = new_distance / np.maximum(old_distance + new_distance, 1e-6)
        # In overlap, favor the closer image but soften cost-heavy pixels.
        weight[overlap] *= np.exp(-difference[overlap] / 80.0)
        return cv2.GaussianBlur(weight, (0, 0), 2)

    def stitch(self, frames: list[np.ndarray], output_size: tuple[int, int] | None = None) -> np.ndarray:
        if len(frames) < 2:
            raise ValueError("RAW classical panorama requires at least two images")
        if any(frame is None or frame.ndim != 3 for frame in frames):
            raise ValueError("All input images must be readable color images")

        # A glob's lexical order is not necessarily capture order.  Build a
        # connected feature graph instead of requiring every adjacent pair to
        # overlap.  Unrelated/blurred frames are safely omitted with a warning.
        features = [self._features(frame) for frame in frames]
        connected: dict[int, np.ndarray] = {0: np.eye(3, dtype=np.float64)}
        remaining = set(range(1, len(frames)))
        while remaining:
            attached = False
            for index in list(remaining):
                connection = self._best_connection(frames, features, connected, index)
                if connection is not None:
                    connected[index] = connection[0]
                    remaining.remove(index)
                    attached = True
            if not attached:
                LOGGER.warning("Skipping %d frame(s) with no reliable feature match: %s", len(remaining), sorted(remaining))
                break
        indices = sorted(connected)
        used_frames = [frames[index] for index in indices]
        transforms = [connected[index] for index in indices]
        translate, size = self._canvas(transforms, used_frames)
        canvas_transform = translate
        canvas_size = size
        if output_size:
            # Do not build a giant full-resolution intermediate only to resize
            # it at the end.  This also makes accidental large extents safe.
            fit = min(1.0, output_size[0] / size[0], output_size[1] / size[1])
            if fit < 1.0:
                canvas_transform = np.array(
                    [[fit, 0, 0], [0, fit, 0], [0, 0, 1]], dtype=np.float64
                ) @ translate
                canvas_size = (
                    max(1, int(np.ceil(size[0] * fit))),
                    max(1, int(np.ceil(size[1] * fit))),
                )
        panorama = np.zeros((canvas_size[1], canvas_size[0], 3), dtype=np.float32)
        valid = np.zeros((canvas_size[1], canvas_size[0]), dtype=bool)
        for frame, transform in zip(used_frames, transforms):
            warped = cv2.warpPerspective(frame, canvas_transform @ transform, canvas_size)
            mask = cv2.warpPerspective(np.ones(frame.shape[:2], np.uint8), canvas_transform @ transform, canvas_size).astype(bool)
            overlap = valid & mask
            if overlap.any():
                weight = self._seam_weight(panorama.astype(np.uint8), warped, overlap)
                panorama[overlap] = (panorama[overlap] * (1 - weight[overlap, None]) + warped[overlap] * weight[overlap, None])
            panorama[mask & ~valid] = warped[mask & ~valid]
            valid |= mask
        result = np.clip(panorama, 0, 255).astype(np.uint8)
        if output_size:
            result = cv2.resize(result, output_size, interpolation=cv2.INTER_LANCZOS4)
        return result


def stitch_raw_files(image_paths: Iterable[str | Path], output_path: str | Path, output_size: tuple[int, int] | None = None) -> np.ndarray:
    """Load, stitch, and write a raw classical panorama."""
    frames = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in image_paths]
    panorama = RawClassicalPanorama().stitch(frames, output_size)
    if not cv2.imwrite(str(output_path), panorama):
        raise OSError(f"Could not write panorama: {output_path}")
    return panorama
