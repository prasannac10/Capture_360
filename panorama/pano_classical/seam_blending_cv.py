"""OpenCV-based seam blending and ghost artifact removal for panoramas."""

import cv2
import numpy as np
from typing import Tuple, Optional, List


class SeamBlendingCV:
    """Perform seam-aware blending for seamless panorama stitching."""

    def __init__(self, blend_type: str = "multiband"):
        """
        Initialize seam blender.

        Args:
            blend_type: "multiband", "feather", or "graph_cut"
        """
        if blend_type not in {"multiband", "feather", "graph_cut"}:
            raise ValueError(f"Unknown blend type: {blend_type}")
        self.blend_type = blend_type

    def find_seam_line(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        overlap_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Find a vertical minimum-cost seam and return left-side ownership.

        Args:
            img1, img2: Input images to blend
            overlap_mask: Binary mask of overlap region

        Returns:
            Seam mask [H, W] (1 = take from img1, 0 = take from img2)
        """
        valid = np.asarray(overlap_mask, dtype=bool)
        if valid.shape != img1.shape[:2] or img1.shape != img2.shape:
            raise ValueError('Images and overlap mask must share a canvas')
        # The legacy pair API supplies overlap only. Require one rectangular
        # overlap rather than inventing validity outside arbitrary holes.
        yy, xx = np.where(valid)
        if not len(xx):
            raise ValueError('Cannot select a seam without overlap')
        x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
        if not valid[y0:y1, x0:x1].all():
            raise ValueError('Irregular overlaps require source validity masks and spherical_composition.seam_masks')
        difference = cv2.absdiff(img1, img2).astype(np.float32)
        if difference.ndim == 3:
            difference = difference.mean(axis=2)
        region = self._compute_minimum_cost_seam(difference[y0:y1, x0:x1], valid[y0:y1, x0:x1])
        ownership = np.zeros(valid.shape, dtype=bool)
        ownership[:, :x0] = True
        ownership[y0:y1, x0:x1] = region
        ownership[:y0, :x0 + np.count_nonzero(region[0])] = True
        ownership[y1:, :x0 + np.count_nonzero(region[-1])] = True
        return ownership

    def _compute_minimum_cost_seam(self, cost_map, mask):
        """Return ownership to the left of a connected vertical minimum-cost path."""
        h, w = cost_map.shape
        if not np.asarray(mask, bool).all():
            raise ValueError('Minimum-cost path requires a rectangular valid overlap')
        cumulative = cost_map[0].astype(np.float32).copy()
        parents = np.zeros((h, w), np.int8)
        for row in range(1, h):
            choices = np.stack((np.r_[np.inf, cumulative[:-1]], cumulative,
                                np.r_[cumulative[1:], np.inf]))
            choice = np.argmin(choices, axis=0)
            parents[row] = choice - 1
            cumulative = cost_map[row] + choices[choice, np.arange(w)]
        column = int(np.argmin(cumulative))
        ownership = np.zeros((h, w), bool)
        for row in range(h - 1, -1, -1):
            ownership[row, :column + 1] = True
            column += int(parents[row, column])
        return ownership

    def blend_images(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        seam_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Blend two images along seam using selected method.

        Args:
            img1, img2: Images to blend
            seam_mask: Seam mask (1 = from img1, 0 = from img2)

        Returns:
            Blended image
        """
        if self.blend_type == "multiband":
            return self._multiband_blend(img1, img2, seam_mask)
        elif self.blend_type == "feather":
            return self._feather_blend(img1, img2, seam_mask)
        elif self.blend_type == "graph_cut":
            return self._graph_cut_blend(img1, img2, seam_mask)
        else:
            return img1 * seam_mask[..., None] + img2 * (~seam_mask)[..., None]

    def _multiband_blend(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        seam_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Multi-band blending (Laplacian pyramid) for seamless blending.

        Args:
            img1, img2: Images to blend
            seam_mask: Seam mask

        Returns:
            Blended image
        """
        num_bands = 4
        pyr1 = self._build_laplacian_pyramid(img1.astype(np.float32), num_bands)
        pyr2 = self._build_laplacian_pyramid(img2.astype(np.float32), num_bands)
        weight = self._create_feather_weight(seam_mask, sigma=20)
        weights = [weight]
        for _ in range(1, num_bands):
            weights.append(cv2.pyrDown(weights[-1]))
        blended = pyr1[-1] * weights[-1][..., None] + pyr2[-1] * (
            1 - weights[-1][..., None]
        )
        for level in range(num_bands - 2, -1, -1):
            size = (pyr1[level].shape[1], pyr1[level].shape[0])
            blended = cv2.pyrUp(blended, dstsize=size)
            blended += pyr1[level] * weights[level][..., None] + pyr2[level] * (
                1 - weights[level][..., None]
            )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _build_laplacian_pyramid(
        self,
        image: np.ndarray,
        levels: int,
    ) -> List[np.ndarray]:
        """Build Laplacian pyramid."""
        gaussian_pyramid = [image]

        for _ in range(levels - 1):
            image = cv2.pyrDown(image)
            gaussian_pyramid.append(image)

        laplacian_pyramid = []
        for i in range(levels - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
            laplacian_pyramid.append(laplacian)

        laplacian_pyramid.append(gaussian_pyramid[-1])
        return laplacian_pyramid

    def _feather_blend(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        seam_mask: np.ndarray,
    ) -> np.ndarray:
        """Simple feather blending at seam boundaries."""
        weight = self._create_feather_weight(seam_mask, sigma=15)
        blended = (
            img1.astype(np.float32) * weight[..., None]
            + img2.astype(np.float32) * (1 - weight)[..., None]
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _create_feather_weight(
        self,
        seam_mask: np.ndarray,
        sigma: float = 15.0,
    ) -> np.ndarray:
        """Create smooth feather weight around seam."""
        # Ownership is a region, not a line. Both constant masks must retain
        # the corresponding source exactly.
        weight = np.asarray(seam_mask, dtype=np.float32)
        return cv2.GaussianBlur(weight, (0, 0), sigma)

    def _graph_cut_blend(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        seam_mask: np.ndarray,
    ) -> np.ndarray:
        """Graph-cut based blending for optimal boundaries."""
        raise ValueError('Graph-cut needs source validity masks; use spherical_composition.seam_masks before multiband blending')


class GhostRemovalCV:
    """Remove ghost artifacts caused by moving objects in panoramic stitching."""

    def __init__(self, threshold_ratio: float = 0.3):
        """
        Initialize ghost remover.

        Args:
            threshold_ratio: Threshold for identifying ghosts (0-1)
        """
        if not 0 <= threshold_ratio <= 1:
            raise ValueError('threshold_ratio must be between zero and one')
        self.threshold_ratio = threshold_ratio

    def detect_ghost_regions(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        overlap_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Detect ghost regions (inconsistent overlaps) in panorama.

        Args:
            img1, img2: Overlapping images
            overlap_mask: Overlap region mask

        Returns:
            Ghost mask [H, W]
        """
        valid = np.asarray(overlap_mask, dtype=bool)
        if valid.shape != img1.shape[:2] or img1.shape != img2.shape:
            raise ValueError('Images and overlap mask must share a canvas')
        if not valid.any():
            return np.zeros(valid.shape, dtype=bool)
        delta = img1.astype(np.float32) - img2.astype(np.float32)
        # Remove a global exposure offset before evaluating disagreement.
        delta -= np.median(delta[valid], axis=0)
        diff = np.abs(delta).mean(axis=2) if delta.ndim == 3 else np.abs(delta)
        median = float(np.median(diff[valid]))
        mad = float(np.median(np.abs(diff[valid] - median)))
        threshold = max(12.0, 255 * self.threshold_ratio, median + 3 * 1.4826 * mad)
        potential_ghost = (diff > threshold) & valid
        return self._filter_connected_components(potential_ghost, valid)

    def _filter_connected_components(
        self,
        mask: np.ndarray,
        valid_region: np.ndarray,
        min_size: int = 50,
    ) -> np.ndarray:
        """Filter connected components by size."""
        num_features, labeled = cv2.connectedComponents(
            mask.astype(np.uint8),
            connectivity=8,
        )

        filtered = np.zeros_like(mask, dtype=bool)

        for i in range(1, num_features):
            component = labeled == i
            if np.sum(component & valid_region) > min_size:
                filtered[component & valid_region] = True

        return filtered

    def remove_ghosts_poisson(
        self,
        panorama: np.ndarray,
        ghost_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Legacy small-defect repair using Telea inpainting (not source deghosting).

        Args:
            panorama: Input panorama
            ghost_mask: Ghost region mask

        Returns:
            Ghost-removed panorama
        """
        if not ghost_mask.any():
            return panorama

        if np.mean(ghost_mask > 0) > 0.01:
            raise ValueError("Large ghost regions need source selection, not inpainting")

        # Dilate mask slightly to ensure smooth transitions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(ghost_mask.astype(np.uint8), kernel, iterations=2)

        # Apply Telea inpainting
        restored = cv2.inpaint(panorama, dilated_mask, 5, cv2.INPAINT_TELEA)

        return restored

    def remove_ghosts_median_filter(
        self,
        images_list: List[np.ndarray],
        ghost_mask_list: List[np.ndarray],
    ) -> np.ndarray:
        """
        Remove ghosts by taking median of multiple exposures.

        Args:
            images_list: List of overlapping images
            ghost_mask_list: List of ghost masks for each image pair

        Returns:
            Ghost-reduced panorama
        """
        if not images_list or len(images_list) != len(ghost_mask_list):
            raise ValueError('Provide one exclusion mask per source image')
        values = np.stack(images_list).astype(np.float32)
        valid = ~np.stack(ghost_mask_list).astype(bool)
        if valid.shape != values.shape[:3]:
            raise ValueError('Ghost masks must match source image dimensions')
        masked = np.ma.array(values, mask=np.broadcast_to(~valid[..., None], values.shape))
        result = np.ma.median(masked, axis=0).filled(0)
        return np.clip(result, 0, 255).astype(np.uint8)

    def detect_motion_blur(
        self,
        image: np.ndarray,
        kernel_size: int = 5,
    ) -> np.ndarray:
        """
        Detect motion blur artifacts in image.

        Args:
            image: Input image
            kernel_size: Size of Sobel kernel

        Returns:
            Motion blur probability map [H, W]
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Compute Laplacian (edge detection)
        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=kernel_size)

        # Low variance in Laplacian indicates blur
        blur_map = cv2.GaussianBlur(np.abs(laplacian), (15, 15), 2.0)
        blur_map = 1.0 - (blur_map / (blur_map.max() + 1e-6))

        return np.clip(blur_map, 0, 1)

    def temporal_consistency_check(
        self,
        prev_panorama: np.ndarray,
        curr_panorama: np.ndarray,
        optical_flow_threshold: float = 5.0,
    ) -> np.ndarray:
        """
        Check temporal consistency to identify moving objects.

        Args:
            prev_panorama: Previous frame panorama
            curr_panorama: Current frame panorama
            optical_flow_threshold: Threshold for detecting motion

        Returns:
            Inconsistency mask (1 = potential ghost)
        """
        # Compute dense optical flow
        if len(prev_panorama.shape) == 3:
            prev_gray = cv2.cvtColor(prev_panorama, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_panorama, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_panorama
            curr_gray = curr_panorama

        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        # Motion magnitude
        magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)

        # Threshold to identify significant motion (potential ghosts)
        inconsistency = magnitude > optical_flow_threshold

        return inconsistency
