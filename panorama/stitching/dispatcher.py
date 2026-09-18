"""Select the AI or classical panorama implementation from YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from .profiles import validate_frame_set


DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the shared panorama configuration."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid panorama configuration: {path}")
    engine = config.get("stitching", {}).get("engine")
    if engine not in {"ai", "classical"}:
        raise ValueError("stitching.engine must be either 'ai' or 'classical'")
    config["_config_path"] = path.resolve()
    return config


def stitch(
    inputs: Iterable[str | Path] | str | Path,
    output_path: str | Path,
    config_path: str | Path | None = None,
    profile: str = "auto",
    **kwargs: Any,
) -> Any:
    """Run the configured stitching implementation.

    ``classical`` expects an iterable of image paths. ``ai`` expects a Capture360
    session directory (containing ``images/``, ``poses.pt``, and ``camera.json``).
    """
    config = load_config(config_path)
    engine = config["stitching"]["engine"]
    if engine == "classical":
        import cv2
        from panorama.pano_classical.opencv_stitcher import stitch_files
        from panorama.pano_classical.corrections import ClassicalCorrectionPipeline

        if isinstance(inputs, (str, Path)):
            raise TypeError("Classical stitching requires an iterable of image paths")
        image_paths = [Path(image) for image in inputs]
        frames = [cv2.imread(str(image)) for image in image_paths]
        if any(frame is None for frame in frames):
            raise ValueError("Could not read one or more input images")
        selected_name, selected_profile = validate_frame_set(
            [(frame.shape[1], frame.shape[0]) for frame in frames], config["input"], profile
        )
        if not selected_profile["min_frames"] <= len(frames) <= selected_profile["max_frames"]:
            raise ValueError(f"{selected_name} requires {selected_profile['min_frames']}..{selected_profile['max_frames']} frames; got {len(frames)}")
        panorama = stitch_files(
            image_paths,
            Path(output_path),
            output_size=(
                int(config["stitching"]["output_width"]),
                int(config["stitching"]["output_height"]),
            ),
            profile=selected_profile,
            **kwargs,
        )
        corrected, _ = ClassicalCorrectionPipeline(config).run(panorama, frames)
        if not cv2.imwrite(str(output_path), corrected):
            raise OSError(f"Could not write corrected panorama: {output_path}")
        return corrected

    from panorama.pano_ai.tiled_inference import run_tiled_inference

    if not isinstance(inputs, (str, Path)):
        raise TypeError("AI stitching requires a Capture360 session directory")
    # Tiled inference reads the native source frames and their camera.json.
    return run_tiled_inference(inputs, config["_config_path"], output_path, profile=profile, **kwargs)
