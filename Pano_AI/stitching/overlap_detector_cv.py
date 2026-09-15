"""OpenCV-based overlap detection for panoramic images."""

import cv2
import numpy as np
from typing import Tuple, List, Dict, Optional


class OverlapDetectorCV:
    """Detect and analyze overlapping regions between views using classical CV techniques."""

    def __init__(self, feature_detector: str = "orb", min_matches: int = 4):
        """
        Initialize overlap detector.
        
        Args:
            feature_detector: "orb", "sift", or "akaze"
            min_matches: Minimum number of feature matches to consider overlap valid
        """
        self.feature_detector = feature_detector
        self.min_matches = min_matches
        self._init_detector()

    def _init_detector(self):
        """Initialize the appropriate feature detector."""
        if self.feature_detector == "orb":
            self.detector = cv2.ORB_create(nfeatures=5000)
        elif self.feature_detector == "sift":
            self.detector = cv2.SIFT_create()
        elif self.feature_detector == "akaze":
            self.detector = cv2.AKAZE_create()
        else:
            raise ValueError(f"Unknown detector: {self.feature_detector}")

        # Initialize matcher
        if self.feature_detector == "sift":
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_pairwise_overlaps(
        self, images: List[np.ndarray]
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Detect overlaps between all image pairs.
        
        Args:
            images: List of input images [H, W, C]
            
        Returns:
            Dictionary mapping (i, j) -> {
                'overlap_percentage': float,
                'match_count': int,
                'homography': np.ndarray,
                'keypoints_i': list,
                'keypoints_j': list,
                'matches': cv2.DMatch list
            }
        """
        n = len(images)
        overlaps = {}

        # Precompute keypoints and descriptors
        kp_desc = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            kp, desc = self.detector.detectAndCompute(gray, None)
            kp_desc.append((kp, desc))

        # Pairwise comparison (only forward direction)
        for i in range(n):
            for j in range(i + 1, n):
                kp_i, desc_i = kp_desc[i]
                kp_j, desc_j = kp_desc[j]

                if desc_i is None or desc_j is None:
                    continue

                # Find matches
                matches = self._find_good_matches(desc_i, desc_j)

                if len(matches) >= self.min_matches:
                    h, overlap_pct, inliers = self._compute_homography_and_overlap(
                        kp_i, kp_j, matches, images[i], images[j]
                    )
                    overlaps[(i, j)] = {
                        'overlap_percentage': overlap_pct,
                        'match_count': len(inliers),
                        'homography': h,
                        'keypoints_i': kp_i,
                        'keypoints_j': kp_j,
                        'matches': inliers,
                        'confidence': len(inliers) / max(len(kp_i), len(kp_j)) if max(len(kp_i), len(kp_j)) > 0 else 0
                    }

        return overlaps

    def _find_good_matches(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Find good matches using Lowe's ratio test.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            
        Returns:
            List of good matches
        """
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        good_matches = []

        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        return good_matches

    def _compute_homography_and_overlap(
        self,
        kp1: List,
        kp2: List,
        matches: List,
        img1: np.ndarray,
        img2: np.ndarray,
    ) -> Tuple[np.ndarray, float, List]:
        """
        Compute homography matrix and estimate overlap percentage.
        
        Args:
            kp1: Keypoints from image 1
            kp2: Keypoints from image 2
            matches: Matched features
            img1: First image
            img2: Second image
            
        Returns:
            (homography_matrix, overlap_percentage, inlier_matches)
        """
        if len(matches) < 4:
            return None, 0.0, []

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        h, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if h is None:
            return None, 0.0, []

        # Filter inliers
        inliers = [m for m, inlier in zip(matches, mask.flatten()) if inlier]

        # Estimate overlap percentage using warped image area
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Warp corners of img1 to img2 space
        corners = np.float32([[[0, 0], [w1, 0], [w1, h1], [0, h1]]])
        warped_corners = cv2.perspectiveTransform(corners, h)

        # Calculate intersection area
        pts_overlap = np.array([
            [0, 0],
            [w2, 0],
            [w2, h2],
            [0, h2]
        ], dtype=np.float32)

        # Use contour intersection
        overlap_pct = self._estimate_overlap_percentage(
            warped_corners[0], pts_overlap, w1, h1, w2, h2
        )

        return h, overlap_pct, inliers

    def _estimate_overlap_percentage(
        self,
        warped_corners: np.ndarray,
        img2_corners: np.ndarray,
        h1: int,
        w1: int,
        h2: int,
        w2: int,
    ) -> float:
        """
        Estimate percentage overlap between two images.
        
        Args:
            warped_corners: Corners of img1 warped to img2 space
            img2_corners: Corners of img2
            h1, w1: Height and width of img1
            h2, w2: Height and width of img2
            
        Returns:
            Overlap percentage (0-100)
        """
        # Create polygon from warped corners
        poly1 = cv2.convexHull(warped_corners.astype(np.int32))
        poly2 = cv2.convexHull(img2_corners.astype(np.int32))

        # Calculate intersection area using masks
        mask1 = np.zeros((h2, w2), dtype=np.uint8)
        mask2 = np.zeros((h2, w2), dtype=np.uint8)

        cv2.fillPoly(mask1, [poly1], 1)
        cv2.fillPoly(mask2, [poly2], 1)

        intersection = np.sum((mask1 & mask2) > 0)
        union = np.sum((mask1 | mask2) > 0)

        if union == 0:
            return 0.0

        overlap_pct = (intersection / union) * 100
        return min(100.0, overlap_pct)

    def detect_overlap_region(
        self, img1: np.ndarray, img2: np.ndarray, h: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the overlap region between two images.
        
        Args:
            img1: First image
            img2: Second image
            h: Homography matrix from img1 to img2
            
        Returns:
            (overlap_region_img1, overlap_region_img2)
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # Warp img1 to img2 space
        warped = cv2.warpPerspective(img1, h, (w2, h2))

        # Create overlap mask
        gray1 = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        overlap_mask = (gray1 > 0) & (gray2 > 0)

        return warped, overlap_mask

    def compute_overlap_statistics(
        self, overlaps: Dict[Tuple[int, int], Dict]
    ) -> Dict:
        """
        Compute statistics from overlap detection results.
        
        Args:
            overlaps: Dictionary from detect_pairwise_overlaps
            
        Returns:
            Dictionary with statistics
        """
        if not overlaps:
            return {
                'total_pairs': 0,
                'overlapping_pairs': 0,
                'mean_overlap': 0.0,
                'mean_confidence': 0.0
            }

        overlap_percentages = [v['overlap_percentage'] for v in overlaps.values()]
        confidences = [v['confidence'] for v in overlaps.values()]

        return {
            'total_pairs': len(overlaps),
            'overlapping_pairs': len([o for o in overlap_percentages if o > 5.0]),
            'mean_overlap': float(np.mean(overlap_percentages)) if overlap_percentages else 0.0,
            'std_overlap': float(np.std(overlap_percentages)) if overlap_percentages else 0.0,
            'mean_confidence': float(np.mean(confidences)) if confidences else 0.0,
            'min_overlap': float(np.min(overlap_percentages)) if overlap_percentages else 0.0,
            'max_overlap': float(np.max(overlap_percentages)) if overlap_percentages else 0.0,
        }
