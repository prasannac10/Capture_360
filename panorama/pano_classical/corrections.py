"""Complete OpenCV correction chain for the classical stitching route."""

from __future__ import annotations

import cv2
import numpy as np

from .overlap_detector_cv import OverlapDetectorCV
from .parallax_correction_cv import ParallaxCorrectionCV
from .seam_blending_cv import GhostRemovalCV, SeamBlendingCV


class ClassicalCorrectionPipeline:
    """Apply every configured correction without loading learned models."""

    def __init__(self, config: dict):
        self.config = config
        params = config["classical_parameters"]
        self.overlap = OverlapDetectorCV(**params["overlap_detection"])
        self.parallax = ParallaxCorrectionCV(
            depth_estimation_method=params["parallax_correction"]["depth_estimation_method"]
        )
        self.ghosts = GhostRemovalCV(params["ghost_removal"]["threshold_ratio"])
        self.seams = SeamBlendingCV(params["seam_blending"]["blend_type"])

    @staticmethod
    def _reference(panorama: np.ndarray, frames: list[np.ndarray]) -> np.ndarray:
        """Bring a captured frame into the panorama working resolution."""
        if not frames:
            return panorama.copy()
        return cv2.resize(frames[0], (panorama.shape[1], panorama.shape[0]), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _remove_glare(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness = lab[..., 0]
        mask = (lightness >= np.percentile(lightness, 99.5)).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def _remove_lens_dots(image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = (gray >= 245).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def _correct_poles(image: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            band = max(1, image.shape[0] // 20)
            mask[:band] = 1
            mask[-band:] = 1
        return cv2.inpaint(image, (mask > 0).astype(np.uint8) * 255, 3, cv2.INPAINT_TELEA)

    @staticmethod
    def _correct_color(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _sharpen(image: np.ndarray) -> np.ndarray:
        return cv2.addWeighted(image, 1.35, cv2.GaussianBlur(image, (0, 0), 1.0), -0.35, 0)

    @staticmethod
    def _correct_parallax_flow(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Use dense Farneback optical flow as the classical flow correction."""
        source = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(source, target, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        return cv2.remap(image, (x + flow[..., 0]).astype(np.float32), (y + flow[..., 1]).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def run(self, panorama: np.ndarray, frames: list[np.ndarray], correction_mask: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        """Run all correction stages and return the corrected panorama plus audit metadata."""
        toggles = {**self.config["correction"]["toggles"], **self.config["advanced_corrections"]["toggles"]}
        out, applied = panorama, []
        if toggles["glare"]: out = self._remove_glare(out); applied.append("glare")
        if toggles["dots"]: out = self._remove_lens_dots(out); applied.append("dots")
        if toggles["nadir_zenith"]: out = self._correct_poles(out, correction_mask); applied.append("nadir_zenith")
        if toggles["color"]: out = self._correct_color(out); applied.append("color")
        reference = self._reference(out, frames)
        if toggles["overlap_detection"]:
            self.overlap.detect_pairwise_overlaps(frames) if len(frames) > 1 else {}
            applied.append("overlap_detection")
        if toggles["parallax"]:
            shift = self.parallax.estimate_parallax_shift(out, reference)
            out = self.parallax.apply_parallax_correction(out, shift); applied.append("parallax")
        if toggles["ghost_removal"]:
            mask = self.ghosts.detect_ghost_regions(out, reference, np.ones(out.shape[:2], dtype=bool))
            out = self.ghosts.remove_ghosts_poisson(out, mask); applied.append("ghost_removal")
        if toggles["seam_blending"]:
            seam = self.seams.find_seam_line(out, reference, np.ones(out.shape[:2], dtype=bool))
            out = self.seams.blend_images(out, reference, seam); applied.append("seam_blending")
        if toggles["parallax_flow"]:
            out = self._correct_parallax_flow(out, reference); applied.append("parallax_flow")
        if toggles["sharpen"]: out = self._sharpen(out); applied.append("sharpen")
        return out, {"engine": "classical", "applied_stages": applied}
