"""Unified integration layer for classical and learned panorama corrections.

This module provides a factory pattern to switch between OpenCV-based classical methods
and trained AI models for panorama enhancement. It serves as the orchestration layer
for the dual-path correction pipeline.
"""

import os
import warnings
from typing import Dict, Optional, List, Tuple
import numpy as np
import cv2
import torch

from pipeline import CorrectionPipeline


class CorrectionMethodFactory:
    """Factory for selecting between classical (OpenCV) and learned (AI) methods."""
    
    @staticmethod
    def create_overlap_detector(config: Dict, device: str = "cpu"):
        """
        Create overlap detector based on config settings.
        
        Args:
            config: Configuration dictionary with 'classical_methods' and 'advanced_corrections'
            device: Device for AI models ("cpu" or "cuda")
            
        Returns:
            Overlap detector instance (classical or learned)
        """
        use_classical = config.get('classical_methods', {}).get('use_classical_overlap_detection', False)
        
        if use_classical:
            from stitching.overlap_detector_cv import OverlapDetectorCV
            feature_detector = config['classical_parameters']['overlap_detection']['feature_detector']
            min_matches = config['classical_parameters']['overlap_detection']['min_matches']
            return OverlapDetectorCV(feature_detector=feature_detector, min_matches=min_matches)
        else:
            from models.advanced_corrections import OverlapDetectionUNet
            model = OverlapDetectionUNet()
            return model.to(device)
    
    @staticmethod
    def create_parallax_corrector(config: Dict, device: str = "cpu"):
        """
        Create parallax corrector based on config settings.
        
        Args:
            config: Configuration dictionary
            device: Device for AI models
            
        Returns:
            Parallax corrector instance (classical or learned)
        """
        use_classical = config.get('classical_methods', {}).get('use_classical_parallax_correction', False)
        
        if use_classical:
            from stitching.parallax_correction_cv import ParallaxCorrectionCV
            method = config['classical_parameters']['parallax_correction']['depth_estimation_method']
            return ParallaxCorrectionCV(depth_estimation_method=method)
        else:
            from models.advanced_corrections import ParallaxCorrectionUNet
            model = ParallaxCorrectionUNet()
            return model.to(device)
    
    @staticmethod
    def create_seam_blender(config: Dict, device: str = "cpu"):
        """
        Create seam blending module based on config settings.
        
        Args:
            config: Configuration dictionary
            device: Device for AI models
            
        Returns:
            Seam blending instance (classical or learned)
        """
        use_classical = config.get('classical_methods', {}).get('use_classical_seam_blending', False)
        
        if use_classical:
            from stitching.seam_blending_cv import SeamBlendingCV
            blend_type = config['classical_parameters']['seam_blending']['blend_type']
            return SeamBlendingCV(blend_type=blend_type)
        else:
            from models.advanced_corrections import SeamBlendingUNet
            model = SeamBlendingUNet()
            return model.to(device)
    
    @staticmethod
    def create_ghost_remover(config: Dict, device: str = "cpu"):
        """
        Create ghost removal module based on config settings.
        
        Args:
            config: Configuration dictionary
            device: Device for AI models
            
        Returns:
            Ghost remover instance (classical or learned)
        """
        use_classical = config.get('classical_methods', {}).get('use_classical_ghost_removal', False)
        
        if use_classical:
            from stitching.seam_blending_cv import GhostRemovalCV
            threshold = config['classical_parameters']['ghost_removal']['threshold_ratio']
            return GhostRemovalCV(threshold_ratio=threshold)
        else:
            from models.advanced_corrections import GhostRemovalUNet
            model = GhostRemovalUNet()
            return model.to(device)


class DualPathPanoramaPipeline:
    """
    Unified panorama processing pipeline supporting both classical and learned paths.
    
    This orchestrates:
    - Classical OpenCV-based corrections
    - Trained AI model corrections
    - Seamless switching between methods via configuration
    """
    
    def __init__(self, config: Dict, device: str = "cpu", allow_untrained: bool = False):
        """
        Initialize dual-path pipeline.
        
        Args:
            config: YAML configuration with correction settings
            device: Device for models ("cpu" or "cuda")
            allow_untrained: Allow untrained model checkpoints
        """
        self.config = config
        self.device = device
        self.allow_untrained = allow_untrained
        
        # Main correction pipeline (always used for baseline corrections)
        self.learned_pipeline = CorrectionPipeline(
            toggles=config['correction']['toggles'],
            checkpoints=config['correction']['checkpoints'],
            device=device,
            allow_untrained=allow_untrained
        )
        
        # Advanced corrections (can be classical or learned)
        self.overlap_detector = CorrectionMethodFactory.create_overlap_detector(config, device)
        self.parallax_corrector = CorrectionMethodFactory.create_parallax_corrector(config, device)
        self.seam_blender = CorrectionMethodFactory.create_seam_blender(config, device)
        self.ghost_remover = CorrectionMethodFactory.create_ghost_remover(config, device)
        
        self.method_summary = self._create_method_summary()
    
    def _create_method_summary(self) -> Dict[str, str]:
        """Create a summary of which methods are being used."""
        return {
            'overlap_detection': 'classical (OpenCV)' if self.config['classical_methods']['use_classical_overlap_detection'] else 'learned (AI)',
            'parallax_correction': 'classical (OpenCV)' if self.config['classical_methods']['use_classical_parallax_correction'] else 'learned (AI)',
            'seam_blending': 'classical (OpenCV)' if self.config['classical_methods']['use_classical_seam_blending'] else 'learned (AI)',
            'ghost_removal': 'classical (OpenCV)' if self.config['classical_methods']['use_classical_ghost_removal'] else 'learned (AI)',
        }
    
    def process_panorama(
        self,
        panorama: np.ndarray,
        images: Optional[List[np.ndarray]] = None,
        camera_params: Optional[np.ndarray] = None,
        poses: Optional[np.ndarray] = None,
        correction_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Process panorama through full dual-path pipeline.
        
        Args:
            panorama: Initial panorama [H, W, 3] uint8
            images: List of input images for advanced corrections
            camera_params: Camera parameters for overlap/parallax detection
            poses: Camera poses for overlap analysis
            correction_mask: Nadir/zenith correction mask
            
        Returns:
            (final_panorama, processing_metadata)
        """
        metadata = {
            'methods_used': self.method_summary,
            'stages': []
        }
        
        # Stage 1: Baseline learned corrections (always run)
        print("[Pipeline] Stage 1: Baseline learned corrections (glare, nadir_zenith, color)")
        panorama = self.learned_pipeline.run(panorama, correction_mask=correction_mask)
        metadata['stages'].append('baseline_corrections')
        
        # Stage 2: Advanced corrections (classical or learned, based on config)
        auxiliary_data = {}
        
        # Overlap detection
        if self.config['advanced_corrections']['toggles'].get('overlap_detection', False):
            print("[Pipeline] Stage 2a: Overlap detection ({})".format(self.method_summary['overlap_detection']))
            overlap_info = self._detect_overlaps(images, camera_params, poses)
            auxiliary_data['overlap_info'] = overlap_info
            metadata['stages'].append('overlap_detection')
        
        # Parallax correction
        if self.config['advanced_corrections']['toggles'].get('parallax', False):
            print("[Pipeline] Stage 2b: Parallax correction ({})".format(self.method_summary['parallax_correction']))
            panorama = self._correct_parallax(panorama, images)
            metadata['stages'].append('parallax_correction')
        
        # Ghost removal
        if self.config['advanced_corrections']['toggles'].get('ghost_removal', False):
            print("[Pipeline] Stage 2c: Ghost removal ({})".format(self.method_summary['ghost_removal']))
            panorama = self._remove_ghosts(panorama, images, auxiliary_data)
            metadata['stages'].append('ghost_removal')
        
        # Seam blending
        if self.config['advanced_corrections']['toggles'].get('seam_blending', False):
            print("[Pipeline] Stage 2d: Seam blending ({})".format(self.method_summary['seam_blending']))
            panorama = self._apply_seam_blending(panorama, images, auxiliary_data)
            metadata['stages'].append('seam_blending')
        
        return panorama, metadata
    
    def _detect_overlaps(
        self,
        images: List[np.ndarray],
        camera_params: Optional[np.ndarray],
        poses: Optional[np.ndarray]
    ) -> Dict:
        """Detect overlaps using configured method."""
        if images is None or len(images) < 2:
            return {}
        
        if isinstance(self.overlap_detector, type) and hasattr(self.overlap_detector, '__name__'):
            # Classical method
            return self.overlap_detector.detect_pairwise_overlaps(images)
        else:
            # Learned method
            return self.overlap_detector.detect_overlaps_batch(images)
    
    def _correct_parallax(self, panorama: np.ndarray, images: Optional[List[np.ndarray]]) -> np.ndarray:
        """Apply parallax correction using configured method."""
        if hasattr(self.parallax_corrector, 'run'):
            # Learned method
            from pipeline import _tensor, _image
            return _image(self.parallax_corrector(_tensor(panorama).to(self.device)))
        else:
            # Classical method - would apply OpenCV-based correction
            return panorama
    
    def _remove_ghosts(
        self,
        panorama: np.ndarray,
        images: Optional[List[np.ndarray]],
        auxiliary_data: Dict
    ) -> np.ndarray:
        """Remove ghost artifacts using configured method."""
        if hasattr(self.ghost_remover, 'run'):
            # Learned method
            from pipeline import _tensor, _image
            return _image(self.ghost_remover(_tensor(panorama).to(self.device)))
        else:
            # Classical method
            if images:
                return self.ghost_remover.remove_ghosts_median_filter(images, [])
        
        return panorama
    
    def _apply_seam_blending(
        self,
        panorama: np.ndarray,
        images: Optional[List[np.ndarray]],
        auxiliary_data: Dict
    ) -> np.ndarray:
        """Apply seam-aware blending using configured method."""
        # Seam blending requires reference images which we approximate from auxiliary data
        if images and len(images) >= 2:
            if hasattr(self.seam_blender, 'forward'):
                # Learned method
                from pipeline import _tensor
                seam_edges = auxiliary_data.get('seam_edges', np.zeros_like(panorama[:, :, 0], dtype=np.float32))
                seam_ref = images[0]
                
                seam_edges_t = torch.from_numpy(seam_edges).float().unsqueeze(0).unsqueeze(0).to(self.device)
                weights = self.seam_blender(_tensor(panorama).to(self.device), _tensor(seam_ref).to(self.device), seam_edges_t)
                
                from pipeline import _image
                blended = _tensor(panorama).to(self.device) * weights + _tensor(seam_ref).to(self.device) * (1 - weights)
                return _image(blended)
            else:
                # Classical method
                overlap_mask = np.ones_like(panorama[:, :, 0], dtype=bool)
                seam_mask = self.seam_blender.find_seam_line(panorama, images[0], overlap_mask)
                return self.seam_blender.blend_images(panorama, images[0], seam_mask)
        
        return panorama
    
    def print_pipeline_config(self):
        """Print current pipeline configuration."""
        print("\n=== Capture360 Dual-Path Pipeline Configuration ===")
        print(f"Device: {self.device}")
        print(f"Allow untrained models: {self.allow_untrained}")
        print("\nBaseline Corrections (Always Learned):")
        for stage, enabled in self.config['correction']['toggles'].items():
            print(f"  - {stage}: {enabled}")
        print("\nAdvanced Corrections (Classical vs Learned):")
        for stage, method in self.method_summary.items():
            print(f"  - {stage}: {method}")
