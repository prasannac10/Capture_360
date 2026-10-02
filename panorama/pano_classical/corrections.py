"""Complete OpenCV correction chain for the classical stitching route."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .overlap_detector_cv import OverlapDetectorCV


class ClassicalCorrectionPipeline:
    """Optional finishing only; geometric correction belongs before composition."""

    def __init__(self, config: dict):
        self.config = config
        params = config["classical_parameters"]
        self.overlap = OverlapDetectorCV(**params["overlap_detection"])

    @staticmethod
    def _inpaint_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Repair small, explicitly identified defects with longitude wrapping."""
        mask = np.asarray(mask)
        if mask.shape != image.shape[:2]:
            raise ValueError('Defect mask must match panorama height and width')
        mask = (mask > 0).astype(np.uint8) * 255
        if np.mean(mask > 0) > 0.01:
            raise ValueError('Inpainting is limited to small defects (at most 1% of pixels)')
        if not mask.any():
            return image.copy()
        pad = min(16, image.shape[1])
        padded = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_WRAP)
        padded_mask = cv2.copyMakeBorder(mask, 0, 0, pad, pad, cv2.BORDER_WRAP)
        result = cv2.inpaint(padded, padded_mask, 3, cv2.INPAINT_TELEA)
        return result[:, pad:-pad].copy()

    @staticmethod
    def _correct_color(image: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            lab[..., 0]
        )
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _sharpen(image: np.ndarray) -> np.ndarray:
        return cv2.addWeighted(
            image, 1.35, cv2.GaussianBlur(image, (0, 0), 1.0), -0.35, 0
        )

    @staticmethod
    def _correct_parallax_flow(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Use dense Farneback optical flow as the classical flow correction."""
        source = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        target = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            target, source, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        return cv2.remap(
            image,
            (x + flow[..., 0]).astype(np.float32),
            (y + flow[..., 1]).astype(np.float32),
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    def run(
        self,
        panorama: np.ndarray,
        frames: list[np.ndarray],
        correction_mask: np.ndarray | None = None,
        output_dir: str | Path | None = None,
        panorama_aligned_frames: list[np.ndarray] | None = None,
        defect_masks: dict[str, np.ndarray] | None = None,
        scene_name: str | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Run all correction stages and return the corrected panorama plus audit metadata."""
        stage_dir = Path(output_dir) if output_dir else None
        if stage_dir:
            stage_dir.mkdir(parents=True, exist_ok=True)

        def save_stage(name: str, image: np.ndarray) -> None:
            if stage_dir and not cv2.imwrite(str(stage_dir / f"{name}.png"), image):
                raise OSError(f"Could not write correction stage: {name}")

        toggles = {
            **self.config["correction"]["toggles"],
            **self.config["advanced_corrections"]["toggles"],
            **self.config.get("classical_finishing", {}),
        }
        out, applied, skipped = panorama, [], {}
        object_report = None
        if toggles.get('object_removal', False):
            from .object_removal import remove_configured_objects
            settings = self.config['classical_parameters'].get('object_removal', {})
            if 'scenes' in settings:
                settings = settings['scenes'].get(scene_name)
            if settings is None:
                skipped['object_removal'] = f'no removal regions configured for scene {scene_name!r}'
            else:
                out, object_mask, object_report = remove_configured_objects(out, settings)
                object_report['scene'] = scene_name
                applied.append('object_removal')
                save_stage('00_object_removal_mask', object_mask)
                save_stage('00_object_removal', out)
        masks = dict(defect_masks or {})
        if correction_mask is not None:
            masks.setdefault('nadir_zenith', correction_mask)
        for stage in ('glare', 'dots', 'nadir_zenith'):
            if not toggles[stage]:
                continue
            if stage not in masks:
                skipped[stage] = 'requires an explicit defect mask; brightness and pole location are not defects'
                continue
            out = self._inpaint_mask(out, masks[stage])
            applied.append(stage)
            save_stage(stage, out)
        if toggles["color"]:
            out = self._correct_color(out)
            applied.append("color")
            save_stage("04_color", out)
        overlap_statistics = None
        if toggles['overlap_detection']:
            if not self.config['classical_methods']['use_classical_overlap_detection']:
                skipped['overlap_detection'] = 'disabled by classical_methods'
            elif len(frames) < 2:
                skipped['overlap_detection'] = 'requires at least two source frames'
            else:
                overlaps = self.overlap.detect_pairwise_overlaps(frames)
                overlap_statistics = self.overlap.compute_overlap_statistics(overlaps)
                applied.append('overlap_detection')
        # A finished blend cannot serve as one of its original source images.
        # Same-sized RGB arrays alone do not supply ownership, validity or
        # occlusion information. Geometry is handled in pose_stitcher instead.
        for stage in ('parallax', 'ghost_removal', 'seam_blending', 'parallax_flow'):
            if toggles[stage]:
                skipped[stage] = 'must run on masked source views before composition; use classical_pose settings'
        if toggles["sharpen"]:
            out = self._sharpen(out)
            applied.append("sharpen")
            save_stage("09_sharpen", out)
        return out, {
            "engine": "classical",
            "applied_stages": applied,
            "skipped_stages": skipped,
            "overlap_statistics": overlap_statistics,
            "object_removal": object_report,
        }
