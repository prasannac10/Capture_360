"""OpenCV-based parallax correction for panoramic images."""

import cv2
import numpy as np
from typing import Tuple, Optional
import warnings


class ParallaxCorrectionCV:
    """Correct parallax distortions in panoramic stitching using classical CV."""

    def __init__(self, depth_estimation_method: str = "stereo_sgbm"):
        """
        Initialize parallax corrector.

        Args:
            depth_estimation_method: "stereo_sgbm" or "disparity_map"
        """
        self.depth_method = depth_estimation_method

    def estimate_parallax_shift(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        h: np.ndarray,
        keypoints1: list,
        keypoints2: list,
        matches: list,
    ) -> np.ndarray:
        """
        Estimate parallax shift vectors using matched features.

        Args:
            img1, img2: Input images
            h: Homography matrix
            keypoints1, keypoints2: Feature keypoints
            matches: Matched features

        Returns:
            Parallax shift map [H, W, 2]
        """
        h_img, w_img = img1.shape[:2]
        parallax_map = np.zeros((h_img, w_img, 2), dtype=np.float32)

        if len(matches) < 4:
            return parallax_map

        # Extract matched point pairs
        src_pts = np.array(
            [keypoints1[m.queryIdx].pt for m in matches], dtype=np.float32
        )
        dst_pts = np.array(
            [keypoints2[m.trainIdx].pt for m in matches], dtype=np.float32
        )

        # Compute parallax vectors at matched points
        parallax_vectors = dst_pts - src_pts

        # Interpolate parallax values across the image using RBF or thin-plate spline
        parallax_map = self._interpolate_parallax(
            src_pts, parallax_vectors, h_img, w_img
        )

        return parallax_map

    def _interpolate_parallax(
        self,
        points: np.ndarray,
        vectors: np.ndarray,
        h: int,
        w: int,
        method: str = "gaussian_blur",
    ) -> np.ndarray:
        """
        Interpolate parallax vectors across the image using Gaussian smoothing.

        Args:
            points: Feature point locations [N, 2]
            vectors: Parallax vectors [N, 2]
            h, w: Output image height and width
            method: Interpolation method

        Returns:
            Interpolated parallax map [H, W, 2]
        """
        parallax_map = np.zeros((h, w, 2), dtype=np.float32)

        # Create sparse maps for x and y parallax components
        sparse_x = np.zeros((h, w), dtype=np.float32)
        sparse_y = np.zeros((h, w), dtype=np.float32)
        count_map = np.zeros((h, w), dtype=np.float32)

        # Accumulate parallax values at feature locations
        for (x, y), (vx, vy) in zip(points, vectors):
            px, py = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
            sparse_x[py, px] += vx
            sparse_y[py, px] += vy
            count_map[py, px] += 1

        # Average overlapping contributions
        mask = count_map > 0
        sparse_x[mask] /= count_map[mask]
        sparse_y[mask] /= count_map[mask]

        # Inpaint and smooth across the entire image
        parallax_map[..., 0] = cv2.inpaint(
            sparse_x.astype(np.uint8),
            (count_map == 0).astype(np.uint8),
            3,
            cv2.INPAINT_TELEA,
        ).astype(np.float32)
        parallax_map[..., 1] = cv2.inpaint(
            sparse_y.astype(np.uint8),
            (count_map == 0).astype(np.uint8),
            3,
            cv2.INPAINT_TELEA,
        ).astype(np.float32)

        # Gaussian smoothing for smoother transitions
        parallax_map[..., 0] = cv2.GaussianBlur(parallax_map[..., 0], (15, 15), 2.0)
        parallax_map[..., 1] = cv2.GaussianBlur(parallax_map[..., 1], (15, 15), 2.0)

        return parallax_map

    def apply_parallax_correction(
        self,
        image: np.ndarray,
        parallax_map: np.ndarray,
    ) -> np.ndarray:
        """
        Apply parallax correction by warping the image.

        Args:
            image: Input image [H, W, C]
            parallax_map: Parallax shift map [H, W, 2]

        Returns:
            Corrected image [H, W, C]
        """
        h, w = image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Apply parallax correction
        map_x = (x + parallax_map[..., 0]).astype(np.float32)
        map_y = (y + parallax_map[..., 1]).astype(np.float32)

        corrected = cv2.remap(
            image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

        return corrected

    def estimate_depth_from_stereo(
        self,
        img_left: np.ndarray,
        img_right: np.ndarray,
        num_disparities: int = 64,
    ) -> np.ndarray:
        """
        Estimate depth map using stereo matching (for depth-based parallax correction).

        Args:
            img_left, img_right: Stereo image pair
            num_disparities: Number of disparity levels

        Returns:
            Depth map
        """
        gray_left = (
            cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            if len(img_left.shape) == 3
            else img_left
        )
        gray_right = (
            cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
            if len(img_right.shape) == 3
            else img_right
        )

        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=5,
            P1=8 * 3 * 5**2,
            P2=32 * 3 * 5**2,
            disp12MaxDiff=1,
            preFilterCap=63,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )

        disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        depth = np.where(disparity > 0, 1.0 / (disparity + 1e-6), 0)

        return depth

    def correct_depth_based_parallax(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        reference_depth: float = 1.0,
    ) -> np.ndarray:
        """
        Correct parallax based on estimated depth values.

        Args:
            image: Input image
            depth_map: Estimated depth map
            reference_depth: Reference depth for normalization

        Returns:
            Parallax-corrected image
        """
        h, w = image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        # Compute correction based on depth variation
        depth_normalized = depth_map / (reference_depth + 1e-6)
        correction_x = (depth_normalized - 1.0) * 5.0  # Scale factor
        correction_y = np.zeros_like(correction_x)

        # Apply correction
        map_x = (x + correction_x).astype(np.float32)
        map_y = (y + correction_y).astype(np.float32)

        corrected = cv2.remap(
            image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT
        )

        return corrected
