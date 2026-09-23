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


def apply_tiled_model(
    image: np.ndarray,
    model: torch.nn.Module,
    tile_size: int,
    overlap: int,
    device: torch.device,
    mask: np.ndarray | None = None,
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
                if mask is not None:
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


class HighResolutionCorrectionPipeline:
    """Run supported, enabled AI corrections and deterministic finishing stages."""

    _MODELS = {
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
        checkpoints = {
            **config.get("correction", {}).get("checkpoints", {}),
            **config.get("advanced_corrections", {}).get("checkpoints", {}),
        }
        allow_untrained = bool(
            config.get("correction", {}).get("allow_untrained", False)
            or config.get("advanced_corrections", {}).get("allow_untrained", False)
        )
        self.models = {}
        for name, model_type in self._MODELS.items():
            if not self.toggles.get(name, False):
                continue
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
            model = model_type().to(self.device)
            if checkpoint and checkpoint.exists():
                state = torch.load(
                    checkpoint, map_location=self.device, weights_only=False
                )
                model.load_state_dict(state.get("model", state), strict=True)
            self.models[name] = model.eval()

    def run(
        self,
        image: np.ndarray,
        mask: np.ndarray | None,
        tile_size: int,
        overlap: int,
        auxiliary: dict[str, np.ndarray] | None = None,
        output_dir: str | Path | None = None,
    ) -> tuple[np.ndarray, dict[str, list[str]]]:
        auxiliary = auxiliary or {}
        stage_dir = Path(output_dir) if output_dir else None
        if stage_dir:
            stage_dir.mkdir(parents=True, exist_ok=True)

        def save_stage(name: str, stage: np.ndarray) -> None:
            if stage_dir:
                Image.fromarray(stage).save(stage_dir / f"{name}.png")

        output, applied = image, []
        for name in ("glare", "nadir_zenith", "color", "parallax", "ghost_removal"):
            if name not in self.models:
                continue
            stage_mask = (
                mask
                if name == "nadir_zenith"
                else auxiliary.get("ghost_mask") if name == "ghost_removal" else None
            )
            output = apply_tiled_model(
                output, self.models[name], tile_size, overlap, self.device, stage_mask
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
            output = remove_lens_dots(output)
            applied.append("dots")
            save_stage(f"{len(applied):02d}_lens_dots", output)
        if self.toggles.get("sharpen", False):
            output = unsharp_mask(output)
            applied.append("sharpen")
            save_stage(f"{len(applied):02d}_sharpen", output)
        return output, {"applied_stages": applied}
