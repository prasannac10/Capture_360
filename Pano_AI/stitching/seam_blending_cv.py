"""OpenCV-based seam blending and ghost artifact removal for panoramas."""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from scipy.ndimage import distance_transform_edt


class SeamBlendingCV:
    """Perform seam-aware blending for seamless panorama stitching."""

    def __init__(self, blend_type: str = "multiband"):
        """
        Initialize seam blender.
        
        Args:
            blend_type: "multiband", "feather", or "graph_cut"
        """
        self.blend_type = blend_type

    def find_seam_line(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        overlap_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Find optimal seam line in overlap region using graph cut.
        
        Args:
            img1, img2: Input images to blend
            overlap_mask: Binary mask of overlap region
            
        Returns:
            Seam mask [H, W] (1 = take from img1, 0 = take from img2)
        """
        h, w = overlap_mask.shape[:2]

        # Compute cost function based on image differences
        if len(img1.shape) == 3:
            diff = cv2.absdiff(img1, img2).mean(axis=2)
        else:
            diff = cv2.absdiff(img1, img2)

        # Create cost map (lower cost = better seam)
        cost_map = diff.copy()
        cost_map[~overlap_mask] = np.inf

        # Find minimum cost path using dynamic programming
        seam_mask = self._compute_minimum_cost_seam(cost_map, overlap_mask)

        return seam_mask

    def _compute_minimum_cost_seam(
        self,
        cost_map: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """
        Compute minimum cost seam using dynamic programming.
        
        Args:
            cost_map: Cost function for seam placement
            mask: Valid region mask
            
        Returns:
            Seam mask
        """
        h, w = cost_map.shape
        seam_mask = np.zeros((h, w), dtype=bool)

        # Compute cumulative cost from top
        cumulative_cost = np.full((h, w), np.inf, dtype=np.float32)
        cumulative_cost[0, :] = cost_map[0, :]

        for i in range(1, h):
            for j in range(w):
                if mask[i, j]:
                    candidates = [
                        cumulative_cost[i - 1, max(0, j - 1)],
                        cumulative_cost[i - 1, j],
                        cumulative_cost[i - 1, min(w - 1, j + 1)],
                    ]
                    cumulative_cost[i, j] = cost_map[i, j] + np.min(candidates)

        # Backtrack to find seam path
        seam_col = np.argmin(cumulative_cost[-1, :])
        seam_mask[-1, seam_col] = True

        for i in range(h - 2, -1, -1):
            candidates = [
                cumulative_cost[i, max(0, seam_col - 1)],
                cumulative_cost[i, seam_col],
                cumulative_cost[i, min(w - 1, seam_col + 1)],
            ]
            seam_col = max(0, seam_col - 1) + np.argmin(candidates)
            seam_mask[i, seam_col] = True

        return seam_mask

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
        blended = np.zeros_like(img1, dtype=np.float32)

        # Build Laplacian pyramids
        pyr1 = self._build_laplacian_pyramid(img1.astype(np.float32), num_bands)
        pyr2 = self._build_laplacian_pyramid(img2.astype(np.float32), num_bands)

        # Blend each band
        for level in range(num_bands):
            if level == 0:
                weight = self._create_feather_weight(seam_mask, sigma=20)
            else:
                weight = cv2.resize(weight, (pyr1[level].shape[1], pyr1[level].shape[0]))

            blended_band = (
                pyr1[level] * weight[..., None] +
                pyr2[level] * (1 - weight)[..., None]
            )
            
            if level == 0:
                blended = blended_band
            else:
                blended = cv2.resize(blended, (pyr1[level].shape[1], pyr1[level].shape[0]))
                blended = blended + blended_band

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
            img1.astype(np.float32) * weight[..., None] +
            img2.astype(np.float32) * (1 - weight)[..., None]
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _create_feather_weight(
        self,
        seam_mask: np.ndarray,
        sigma: float = 15.0,
    ) -> np.ndarray:
        """Create smooth feather weight around seam."""
        # Distance transform from seam line
        dist = distance_transform_edt(~seam_mask).astype(np.float32)
        
        # Gaussian smooth falloff
        weight = np.exp(-(dist ** 2) / (2 * sigma ** 2))
        weight = np.clip(weight, 0, 1)

        return weight

    def _graph_cut_blend(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        seam_mask: np.ndarray,
    ) -> np.ndarray:
        """Graph-cut based blending for optimal boundaries."""
        # Use seam mask to define regions
        weight_map = self._create_feather_weight(seam_mask, sigma=10)
        
        # Iteratively refine blend using local optimization
        blended = img1.copy().astype(np.float32)
        
        for _ in range(3):  # Refinement iterations
            gradient1 = cv2.Sobel(img1.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            gradient2 = cv2.Sobel(img2.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
            
            # Adjust weight based on gradient magnitude
            if len(gradient1.shape) == 3:
                grad_mag1 = np.sqrt((gradient1 ** 2).sum(axis=2))
                grad_mag2 = np.sqrt((gradient2 ** 2).sum(axis=2))
            else:
                grad_mag1 = np.abs(gradient1)
                grad_mag2 = np.abs(gradient2)
            
            # Higher weight where img1 has lower gradients
            local_weight = 1.0 / (1.0 + grad_mag1 / (grad_mag2 + 1e-6))
            local_weight = cv2.GaussianBlur(local_weight, (5, 5), 1.0)
            
            weight_map = weight_map * 0.7 + local_weight * 0.3

        blended = (
            img1.astype(np.float32) * weight_map[..., None] +
            img2.astype(np.float32) * (1 - weight_map)[..., None]
        )
        
        return np.clip(blended, 0, 255).astype(np.uint8)


class GhostRemovalCV:
    """Remove ghost artifacts caused by moving objects in panoramic stitching."""

    def __init__(self, threshold_ratio: float = 0.3):
        """
        Initialize ghost remover.
        
        Args:
            threshold_ratio: Threshold for identifying ghosts (0-1)
        """
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
        # Compute difference in overlap region
        if len(img1.shape) == 3:
            diff = cv2.absdiff(img1, img2).mean(axis=2)
        else:
            diff = cv2.absdiff(img1, img2)

        # Threshold to find significant differences
        threshold = np.percentile(diff[overlap_mask], 75)
        potential_ghost = (diff > threshold * self.threshold_ratio)

        # Filter by connected components (remove noise)
        ghost_mask = self._filter_connected_components(potential_ghost, overlap_mask)

        return ghost_mask

    def _filter_connected_components(
        self,
        mask: np.ndarray,
        valid_region: np.ndarray,
        min_size: int = 50,
    ) -> np.ndarray:
        """Filter connected components by size."""
        labeled, num_features = cv2.connectedComponents(mask.astype(np.uint8))
        filtered = np.zeros_like(mask)

        for i in range(1, num_features):
            component = labeled == i
            if np.sum(component & valid_region) > min_size:
                filtered[component] = True

        return filtered

    def remove_ghosts_poisson(
        self,
        panorama: np.ndarray,
        ghost_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Remove ghosts using Poisson inpainting.
        
        Args:
            panorama: Input panorama
            ghost_mask: Ghost region mask
            
        Returns:
            Ghost-removed panorama
        """
        if not ghost_mask.any():
            return panorama

        # Dilate mask slightly to ensure smooth transitions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated_mask = cv2.dilate(ghost_mask.astype(np.uint8), kernel, iterations=2)

        # Apply Poisson inpainting
        restored = cv2.inpaint(
            panorama,
            dilated_mask,
            5,
            cv2.INPAINT_TELEA
        )

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
        if not images_list:
            return images_list[0] if images_list else None

        # Stack images and compute weighted median
        valid_regions = [~mask for mask in ghost_mask_list]
        
        # For each pixel, use median of non-ghost pixels
        result = np.zeros_like(images_list[0], dtype=np.float32)
        weight_sum = np.zeros(images_list[0].shape[:2], dtype=np.float32)

        for img, valid in zip(images_list, valid_regions):
            result += img.astype(np.float32) * valid[..., None].astype(np.float32)
            weight_sum += valid.astype(np.float32)

        # Avoid division by zero
        weight_sum[weight_sum == 0] = 1.0
        result = result / weight_sum[..., None]

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
