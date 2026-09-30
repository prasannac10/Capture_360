"""Memory-bounded correction stages for native-resolution tiled panoramas."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from .models.advanced_corrections import GhostRemovalUNet, ParallaxCorrectionUNet
from .models.color_enhance import ColorEnhancementUNet
from .models.combined_restoration import CombinedRestorationUNet
from .models.glare_removal import GlareRemovalUNet
from .models.lens_dots import remove_lens_dots
from .models.nadir_zenith import NadirZenithInpainter
from .models.sharpen import unsharp_mask


def _tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0).div(255)


def _image(tensor: torch.Tensor) -> np.ndarray:
    return (tensor[0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1) * 255).astype(
        np.uint8
    )


def _apply_tiled_model(
    image: np.ndarray,
    model: torch.nn.Module,
    tile_size: int,
    overlap: int,
    device: torch.device,
    mask: np.ndarray | None = None,
    mask_as_input=True,
) -> np.ndarray:
    """Apply a single-image correction model to overlapping native-resolution tiles."""
    height, width = image.shape[:2]
    step = tile_size - overlap
    ys = list(range(0, max(1, height - tile_size + 1), step))
    xs = list(range(0, max(1, width - tile_size + 1), step))
    if height > tile_size and ys[-1] != height - tile_size:
        ys.append(height - tile_size)
    if width > tile_size and xs[-1] != width - tile_size:
        xs.append(width - tile_size)
    output = np.zeros_like(image, dtype=np.float32)
    weights = np.zeros((height, width, 1), dtype=np.float32)
    window = np.maximum(
        cv2.createHanningWindow((tile_size, tile_size), cv2.CV_32F), 1e-3
    )[..., None]
    with torch.inference_mode():
        for y in ys:
            for x in xs:
                crop = image[
                    y : min(y + tile_size, height), x : min(x + tile_size, width)
                ]
                crop_height, crop_width = crop.shape[:2]
                padded = cv2.copyMakeBorder(
                    crop,
                    0,
                    tile_size - crop_height,
                    0,
                    tile_size - crop_width,
                    cv2.BORDER_REFLECT_101,
                )
                arguments = [_tensor(padded).to(device)]
                if mask is not None and mask_as_input:
                    mask_crop = mask[y : y + crop_height, x : x + crop_width]
                    mask_crop = cv2.copyMakeBorder(
                        mask_crop,
                        0,
                        tile_size - crop_height,
                        0,
                        tile_size - crop_width,
                        cv2.BORDER_CONSTANT,
                    )
                    arguments.append(
                        torch.from_numpy(mask_crop)
                        .float()
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .to(device)
                    )
                predicted = _image(model(*arguments))[:crop_height, :crop_width].astype(
                    np.float32
                )
                weight = window[:crop_height, :crop_width]
                output[y : y + crop_height, x : x + crop_width] += predicted * weight
                weights[y : y + crop_height, x : x + crop_width] += weight
    return np.clip(output / np.maximum(weights, 1e-6), 0, 255).astype(np.uint8)


def validate_mask(mask, shape):
    if mask is None:
        raise ValueError('An explicit defect mask is required for masked correction')
    mask = np.asarray(mask, dtype=np.float32)
    if mask.shape != shape or not np.isfinite(mask).all() or np.any((mask < 0) | (mask > 1)):
        raise ValueError('Correction mask must match image height/width and contain finite values in [0,1]')
    return mask


def apply_tiled_model(image, model, tile_size, overlap, device, mask=None, mask_as_input=True):
    if tile_size < 8 or not 0 <= overlap < tile_size:
        raise ValueError('Require tile_size >= 8 and 0 <= overlap < tile_size')
    if mask is not None:
        mask = validate_mask(mask, image.shape[:2])
        if not mask.any():
            return image.copy()
    # Give both ends of the panorama real neighbouring longitude context.
    pad = min(image.shape[1], max(1, overlap // 2))
    padded = np.pad(image, ((0, 0), (pad, pad), (0, 0)), mode='wrap')
    padded_mask = np.pad(mask, ((0, 0), (pad, pad)), mode='wrap') if mask is not None else None
    result = _apply_tiled_model(padded, model, tile_size, overlap, device, padded_mask, mask_as_input)[:, pad:-pad]
    if mask is not None:
        result = np.rint(image * (1 - mask[..., None]) + result * mask[..., None]).clip(0, 255).astype(np.uint8)
    return result


class HighResolutionCorrectionPipeline:
    """Run supported, enabled AI corrections and deterministic finishing stages."""

    _MODELS = {
        "combined": CombinedRestorationUNet,
        "glare": GlareRemovalUNet,
        "nadir_zenith": NadirZenithInpainter,
        "color": ColorEnhancementUNet,
        "parallax": ParallaxCorrectionUNet,
        "ghost_removal": GhostRemovalUNet,
    }

    def __init__(
        self,
        config: dict[str, Any],
        config_path: str | Path,
        device: str | torch.device = "cpu",
    ):
        self.config = config
        self.config_path = Path(config_path).resolve()
        self.device = torch.device(device)
        self.toggles = {
            **config.get("correction", {}).get("toggles", {}),
            **config.get("advanced_corrections", {}).get("toggles", {}),
        }
        if self.toggles.get('combined') and any(value for key, value in self.toggles.items() if key != 'combined'):
            raise ValueError('Combined restoration replaces other post-blend corrections; enable it alone')
        unsupported = [name for name in ('seam_blending', 'overlap_detection', 'parallax_flow')
                       if self.toggles.get(name, False)]
        if unsupported:
            raise NotImplementedError(f'Pairwise tiled AI corrections are not implemented: {unsupported}')
        checkpoints = {
            **config.get("correction", {}).get("checkpoints", {}),
            **config.get("advanced_corrections", {}).get("checkpoints", {}),
        }
        self.models = {}
        for name, model_type in self._MODELS.items():
            if not self.toggles.get(name, False):
                continue
            section = 'advanced_corrections' if name in ('parallax', 'ghost_removal') else 'correction'
            allow_untrained = bool(config.get(section, {}).get('allow_untrained', False))
            value = checkpoints.get(name)
            checkpoint = (
                self.config_path.parent / value
                if value and not Path(value).is_absolute()
                else Path(value) if value else None
            )
            if checkpoint is None or not checkpoint.exists():
                if not allow_untrained:
                    raise FileNotFoundError(
                        f"Enabled correction checkpoint missing: {name}: {checkpoint}"
                    )
            state = None
            if checkpoint and checkpoint.exists():
                state = torch.load(
                    checkpoint, map_location=self.device, weights_only=False
                )
                if state.get('task', name) != name:
                    raise ValueError(f'Checkpoint task does not match {name}')
                if name == 'combined' and state.get('contract') != 'paired_panorama_restoration_v1':
                    raise ValueError('Combined restoration requires its own paired-panorama checkpoint')
            channel_key = 'base_channels' if name in ('parallax', 'ghost_removal') else 'channels'
            model = model_type(**{channel_key: state.get('channels', 32) if state else 32}).to(self.device)
            if state is not None:
                section_cfg = config.get(section, {})
                use_ema = section_cfg.get('use_ema', True)
                weights = state.get('ema') if use_ema else None
                model.load_state_dict(weights if weights is not None else state.get('model', state), strict=True)
            self.models[name] = model.eval()

    def run(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        tile_size: int,
        overlap: int,
        auxiliary: dict[str, np.ndarray] | None = None,
        output_dir: str | Path | None = None,
        defect_masks: dict[str, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, dict[str, list[str]]]:
        auxiliary = auxiliary or {}
        defects = defect_masks or {}
        if 'glare' in self.models:
            validate_mask(defects.get('glare'), image.shape[:2])
        if self.toggles.get('dots', False):
            validate_mask(defects.get('lens_dots'), image.shape[:2])
        if 'nadir_zenith' in self.models:
            mask = validate_mask(mask, image.shape[:2])
        if 'ghost_removal' in self.models:
            auxiliary = dict(auxiliary)
            auxiliary['ghost_mask'] = validate_mask(auxiliary.get('ghost_mask'), image.shape[:2])
        stage_dir = Path(output_dir) if output_dir else None
        if stage_dir:
            stage_dir.mkdir(parents=True, exist_ok=True)

        def save_stage(name: str, stage: np.ndarray) -> None:
            if stage_dir:
                Image.fromarray(stage).save(stage_dir / f"{name}.png")

        output, applied = image, []
        for name in ("combined", "glare", "nadir_zenith", "color", "parallax", "ghost_removal"):
            if name not in self.models:
                continue
            stage_mask = (
                mask
                if name == "nadir_zenith"
                else auxiliary.get("ghost_mask") if name == "ghost_removal"
                else defects.get('glare') if name == 'glare' else None
            )
            output = apply_tiled_model(
                output, self.models[name], tile_size, overlap, self.device, stage_mask,
                mask_as_input=bool(getattr(self.models[name], 'mask_channels', 0))
            )
            applied.append(name)
            save_stage(f"{len(applied):02d}_{name}", output)
        if (
            self.toggles.get("seam_blending")
            or self.toggles.get("overlap_detection")
            or self.toggles.get("parallax_flow")
        ):
            raise NotImplementedError(
                "Pairwise AI corrections require panorama-aligned reference/mask contracts; do not enable them until those assets and tiled pairwise implementations are provided."
            )
        if self.toggles.get("dots", False):
            selected = (defects['lens_dots'] > 0).astype(np.uint8) * 255
            if np.mean(selected > 0) > .01:
                raise ValueError('Lens-dot inpainting is limited to 1% of pixels')
            padded = cv2.copyMakeBorder(output, 0, 0, 16, 16, cv2.BORDER_WRAP)
            padded_mask = cv2.copyMakeBorder(selected, 0, 0, 16, 16, cv2.BORDER_WRAP)
            output = cv2.inpaint(padded, padded_mask, 3, cv2.INPAINT_TELEA)[:, 16:-16]
            applied.append("dots")
            save_stage(f"{len(applied):02d}_lens_dots", output)
        if self.toggles.get("sharpen", False):
            output = unsharp_mask(output)
            applied.append("sharpen")
            save_stage(f"{len(applied):02d}_sharpen", output)
        return output, {"applied_stages": applied}
