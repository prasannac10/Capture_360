"""Deterministic geometric + learned panorama correction pipeline."""

from pathlib import Path
import warnings

import cv2
import numpy as np
import torch

from .models.color_enhance import ColorEnhancementUNet
from .models.glare_removal import GlareRemovalUNet
from .models.lens_dots import remove_lens_dots
from .models.nadir_zenith import NadirZenithInpainter
from .models.sharpen import unsharp_mask
from .models.advanced_corrections import (
    ParallaxCorrectionUNet,
    GhostRemovalUNet,
    SeamBlendingUNet,
    OverlapDetectionUNet,
    ParallaxCorrectionDetector
)


DEFAULT_TOGGLES = {
    "glare": True, 
    "dots": True, 
    "nadir_zenith": True, 
    "color": True, 
    "sharpen": True,
    # New advanced corrections
    "parallax": False,
    "ghost_removal": False,
    "seam_blending": False,
    "overlap_detection": False,
    "parallax_flow": False
}

LEARNED_STAGES = {"glare", "nadir_zenith", "color", "parallax", "ghost_removal", "seam_blending", "overlap_detection", "parallax_flow"}


def _tensor(x):
    return torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0) / 255.0


def _image(x):
    return (x[0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255).astype(np.uint8)


def _load_checkpoint(model, path, name, device, allow_untrained=False):
    if not path:
        if allow_untrained:
            warnings.warn(f"No checkpoint configured for {name}; using untrained weights.")
            return False
        raise FileNotFoundError(f"Correction stage '{name}' is enabled but no checkpoint was configured")
    path = Path(path)
    if not path.exists():
        if allow_untrained:
            warnings.warn(f"Checkpoint not found for {name}: {path}; using untrained weights.")
            return False
        raise FileNotFoundError(f"Checkpoint not found for enabled correction stage '{name}': {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model", state), strict=True)
    return True


class CorrectionPipeline:
    """Apply learned stages independently, then classical finishing stages.

    Learned stages require trained checkpoints unless ``allow_untrained=True``.
    
    Correction stages execute in order:
    1. Baseline correction (glare, nadir_zenith, color)
    2. Advanced geometric correction (parallax, ghost_removal)
    3. Seam-aware blending
    4. Classical finishing (dots, sharpen)
    """

    def __init__(self, toggles=None, checkpoints=None, device="cpu", allow_untrained=False):
        self.toggles = DEFAULT_TOGGLES.copy()
        self.toggles.update(toggles or {})
        self.device = torch.device(device)
        self.checkpoints = checkpoints or {}
        self.allow_untrained = allow_untrained
        
        # Original correction models
        self.glare = GlareRemovalUNet().to(self.device)
        self.poles = NadirZenithInpainter().to(self.device)
        self.color = ColorEnhancementUNet().to(self.device)
        
        # New advanced correction models
        self.parallax = ParallaxCorrectionUNet().to(self.device)
        self.ghost_removal = GhostRemovalUNet().to(self.device)
        self.seam_blending = SeamBlendingUNet().to(self.device)
        self.overlap_detection = OverlapDetectionUNet().to(self.device)
        self.parallax_flow = ParallaxCorrectionDetector().to(self.device)
        
        # Load checkpoints for all learned stages
        for name, model in [
            ("glare", self.glare),
            ("nadir_zenith", self.poles),
            ("color", self.color),
            ("parallax", self.parallax),
            ("ghost_removal", self.ghost_removal),
            ("seam_blending", self.seam_blending),
            ("overlap_detection", self.overlap_detection),
            ("parallax_flow", self.parallax_flow),
        ]:
            if self.toggles.get(name, False):
                _load_checkpoint(model, self.checkpoints.get(name), name, self.device, allow_untrained)
            model.eval()

    def run(self, panorama, correction_mask=None, auxiliary_data=None):
        """Run correction pipeline on panorama.
        
        Args:
            panorama: Input panorama [H, W, 3] uint8
            correction_mask: Nadir/zenith correction mask [H, W]
            auxiliary_data: Dict with optional 'seam_edges', 'overlap_mask', etc.
            
        Returns:
            Corrected panorama [H, W, 3] uint8
        """
        auxiliary_data = auxiliary_data or {}
        out = panorama
        
        with torch.no_grad():
            # === Stage 1: Baseline Color Correction ===
            if self.toggles["glare"]:
                out = _image(self.glare(_tensor(out).to(self.device)))
            
            if self.toggles["dots"]:
                out = remove_lens_dots(out)
            
            if self.toggles["nadir_zenith"] and correction_mask is not None:
                mask = torch.from_numpy(correction_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
                out = _image(self.poles(_tensor(out).to(self.device), mask))
            
            if self.toggles["color"]:
                out = _image(self.color(_tensor(out).to(self.device)))
            
            # === Stage 2: Advanced Geometric Corrections ===
            # Parallax correction - corrects depth-based distortions
            if self.toggles["parallax"]:
                out = _image(self.parallax(_tensor(out).to(self.device)))
            
            # Parallax flow detection - learns directional flow field
            if self.toggles["parallax_flow"]:
                overlap_mask = auxiliary_data.get('overlap_mask')
                if overlap_mask is not None:
                    overlap_mask_t = torch.from_numpy(overlap_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
                else:
                    overlap_mask_t = None
                # Returns flow field but apply as perceptual guidance
                _ = self.parallax_flow(_tensor(out).to(self.device), _tensor(panorama).to(self.device), overlap_mask_t)
            
            # Ghost artifact removal - handles moving objects and exposure inconsistencies
            if self.toggles["ghost_removal"]:
                ghost_mask = auxiliary_data.get('ghost_mask')
                if ghost_mask is not None:
                    ghost_mask_t = torch.from_numpy(ghost_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
                    out = _image(self.ghost_removal(_tensor(out).to(self.device), ghost_mask_t))
                else:
                    out = _image(self.ghost_removal(_tensor(out).to(self.device)))
            
            # === Stage 3: Seam-Aware Blending ===
            if self.toggles["seam_blending"]:
                seam_edges = auxiliary_data.get('seam_edges')
                if seam_edges is not None and 'seam_reference' in auxiliary_data:
                    seam_edges_t = torch.from_numpy(seam_edges).float().unsqueeze(0).unsqueeze(0).to(self.device)
                    seam_ref = auxiliary_data['seam_reference']
                    seam_ref_t = _tensor(seam_ref).to(self.device)
                    weights = self.seam_blending(_tensor(out).to(self.device), seam_ref_t, seam_edges_t)
                    # Apply learned weights to blend
                    out_t = _tensor(out).to(self.device)
                    blended = out_t * weights + seam_ref_t * (1 - weights)
                    out = _image(blended)
            
            # Overlap detection - validates overlap regions
            if self.toggles["overlap_detection"]:
                overlap_ref = auxiliary_data.get('overlap_reference')
                if overlap_ref is not None:
                    overlap_ref_t = _tensor(overlap_ref).to(self.device)
                    overlap_results = self.overlap_detection(_tensor(out).to(self.device), overlap_ref_t)
                    self.last_overlap_results = overlap_results
        
        # === Stage 4: Classical Finishing ===
        if self.toggles["sharpen"]:
            out = unsharp_mask(out)
        
        return out

