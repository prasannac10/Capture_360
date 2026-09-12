"""Deterministic geometric + learned panorama correction pipeline."""

from pathlib import Path
import warnings

import cv2
import numpy as np
import torch

from models.color_enhance import ColorEnhancementUNet
from models.glare_removal import GlareRemovalUNet
from models.lens_dots import remove_lens_dots
from models.nadir_zenith import NadirZenithInpainter
from models.sharpen import unsharp_mask


DEFAULT_TOGGLES = {"glare": True, "dots": True, "nadir_zenith": True, "color": True, "sharpen": True}
LEARNED_STAGES = {"glare", "nadir_zenith", "color"}


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
    ``correction_mask`` uses 1 for pixels that should be repaired by the pole
    inpainting model.
    """

    def __init__(self, toggles=None, checkpoints=None, device="cpu", allow_untrained=False):
        self.toggles = DEFAULT_TOGGLES.copy()
        self.toggles.update(toggles or {})
        self.device = torch.device(device)
        self.checkpoints = checkpoints or {}
        self.allow_untrained = allow_untrained
        self.glare = GlareRemovalUNet().to(self.device)
        self.poles = NadirZenithInpainter().to(self.device)
        self.color = ColorEnhancementUNet().to(self.device)
        for name, model in (("glare", self.glare), ("nadir_zenith", self.poles), ("color", self.color)):
            if self.toggles[name]:
                _load_checkpoint(model, self.checkpoints.get(name), name, self.device, allow_untrained)
            model.eval()

    def run(self, panorama, correction_mask=None):
        out = panorama
        with torch.no_grad():
            if self.toggles["glare"]:
                out = _image(self.glare(_tensor(out).to(self.device)))
            if self.toggles["dots"]:
                out = remove_lens_dots(out)
            if self.toggles["nadir_zenith"] and correction_mask is not None:
                mask = torch.from_numpy(correction_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
                out = _image(self.poles(_tensor(out).to(self.device), mask))
            if self.toggles["color"]:
                out = _image(self.color(_tensor(out).to(self.device)))
        if self.toggles["sharpen"]:
            out = unsharp_mask(out)
        return out


def run_pipeline(image_paths, output_path, toggles=None, checkpoints=None, device="cpu", allow_untrained=False):
    from stitching.opencv_stitcher import stitch_fisheye_files
    baseline = stitch_fisheye_files([Path(p) for p in image_paths], output_path)
    return CorrectionPipeline(toggles, checkpoints, device, allow_untrained).run(baseline)
