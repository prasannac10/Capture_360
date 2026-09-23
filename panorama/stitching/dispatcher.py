"""Select the AI or classical panorama implementation from YAML configuration."""

from __future__ import annotations

import json
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
    engine: str | None = None,
    **kwargs: Any,
) -> Any:
    """Run the configured stitching implementation.

    ``classical`` expects an iterable of image paths. ``ai`` expects a Capture360
    session directory (containing ``images/``, ``poses.pt``, and ``camera.json``).
    """
    config = load_config(config_path)
    selected_engine = engine or config["stitching"]["engine"]
    if selected_engine not in {"ai", "classical"}:
        raise ValueError("engine must be either 'ai' or 'classical'")
    if selected_engine == "classical":
        import cv2
        from panorama.pano_classical.opencv_stitcher import stitch_files
        from panorama.pano_classical.corrections import ClassicalCorrectionPipeline

        if isinstance(inputs, (str, Path)):
            raise TypeError("Classical stitching requires an iterable of image paths")
        requested_output = Path(output_path)
        # The CLI documents an output directory, but accepting a filename is
        # friendlier for direct command-line use.  Intermediate raw/metadata
        # files remain beside that requested image.
        output_dir = requested_output.parent if requested_output.suffix else requested_output
        final_output = requested_output if requested_output.suffix else output_dir / "final_panorama.png"
        if requested_output.suffix and requested_output.is_dir():
            raise IsADirectoryError(
                f"Output image path is an existing directory: {requested_output}. "
                "Remove or rename that directory, then rerun."
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        image_paths = [Path(image) for image in inputs]
        dimensions = []
        for image in image_paths:
            frame = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError(f"Could not read input image: {image}")
            dimensions.append((frame.shape[1], frame.shape[0]))
            del frame
        selected_name, selected_profile = validate_frame_set(
            dimensions,
            config["input"],
            profile,
        )
        if (
            not selected_profile["min_frames"]
            <= len(dimensions)
            <= selected_profile["max_frames"]
        ):
            raise ValueError(
                f"{selected_name} requires {selected_profile['min_frames']}..{selected_profile['max_frames']} frames; got {len(dimensions)}"
            )
        panorama = stitch_files(
            image_paths,
            output_dir / "raw_classical_panorama.png",
            output_size=(
                int(config["stitching"]["output_width"]),
                int(config["stitching"]["output_height"]),
            ),
            profile=selected_profile,
            **kwargs,
        )
        corrected, metadata = ClassicalCorrectionPipeline(config).run(
            panorama,
            [],
            output_dir=output_dir,
        )
        if not cv2.imwrite(str(final_output), corrected):
            raise OSError(f"Could not write final panorama: {final_output}")
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return corrected

    from panorama.pano_ai.tiled_inference import run_tiled_inference

    if not isinstance(inputs, (str, Path)):
        raise TypeError("AI stitching requires a Capture360 session directory")
    # Tiled inference reads the native source frames and their camera.json.
    return run_tiled_inference(
        inputs, config["_config_path"], output_path, profile=profile, **kwargs
    )
