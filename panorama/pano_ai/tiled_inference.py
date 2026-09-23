"""Authoritative native-resolution tiled AI inference entry point."""

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from panorama.stitching.profiles import validate_frame_set
from .data.tile_dataset import VariableTilePanoramaDataset, iter_tile_batches
from .highres_correction import HighResolutionCorrectionPipeline
from .models.panorama_model import PanoramaModel


def _pole_mask(height, width):
    mask = np.zeros((height, width), dtype=np.float32)
    band = max(1, height // 20)
    mask[:band] = mask[-band:] = 1.0
    return mask


def run_tiled_inference(
    scene, config_path, output_dir=None, checkpoint=None, profile="auto"
):
    config_path = Path(config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg, inference_cfg = config["model"], config["inference"]
    device, scene = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        Path(scene).resolve(),
    )
    dataset = VariableTilePanoramaDataset(
        scene.parent,
        False,
        model_cfg["tile_size"],
        model_cfg["tile_overlap"],
        config["input"]["min_frames"],
        config["input"]["max_frames"],
    )
    sample = dataset[
        next(
            index
            for index, path in enumerate(dataset.scenes)
            if path.resolve() == scene
        )
    ]
    profile_name, capture_profile = validate_frame_set(
        [tuple(size.int().tolist()) for size in sample["image_size"]],
        config["input"],
        profile,
    )
    expected_projection = 0.0 if capture_profile["projection"] == "fisheye_180" else 1.0
    if not bool(torch.all(sample["camera_params"][:, 4] == expected_projection)):
        raise ValueError(
            f"{profile_name} camera.json projection does not match its configured profile"
        )
    checkpoint_path = (
        Path(checkpoint)
        if checkpoint
        else config_path.parent / inference_cfg["checkpoint"]
    )
    state = torch.load(
        checkpoint_path.resolve(), map_location=device, weights_only=False
    )
    model = PanoramaModel(
        model_cfg["feature_dim"],
        (model_cfg["pano_feature_height"], model_cfg["pano_feature_width"]),
        (model_cfg["output_height"], model_cfg["output_width"]),
        model_cfg["tile_size"],
        model_cfg["encoder"]["backbone"],
        model_cfg["encoder"]["pretrained"],
        model_cfg["attention"]["heads"],
        model_cfg["attention"].get("layers", 2),
        inference_cfg.get("output_tile", 1024),
    ).to(device)
    model.load_state_dict(state.get("model", state), strict=True)
    model.eval()
    batch_size = int(inference_cfg.get("tile_batch_size", 4))
    with torch.inference_mode():
        prediction = model.forward_scene(
            lambda: iter_tile_batches(
                sample, model_cfg["tile_size"], model_cfg["tile_overlap"], batch_size
            ),
            sample["image_size"],
            sample["camera_params"],
            sample["poses"],
            batch_size,
        )[0]
    initial = (
        (prediction.permute(1, 2, 0).cpu().numpy().clip(0, 1) * 255)
        .round()
        .astype(np.uint8)
    )
    output = (
        Path(output_dir or config_path.parent / inference_cfg["output_dir"])
        / sample["scene"]
    )
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(initial).save(output / "initial_panorama.png")
    Image.fromarray(initial).save(
        output / "initial_panorama.tiff", compression="tiff_deflate"
    )
    mask_path = scene / "correction_mask.png"
    correction_mask = (
        np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255
        if mask_path.exists()
        else _pole_mask(*initial.shape[:2])
    )
    corrected, corrections = HighResolutionCorrectionPipeline(
        config, config_path, device
    ).run(
        initial,
        correction_mask,
        model_cfg["tile_size"],
        model_cfg["tile_overlap"],
        output_dir=output,
    )
    Image.fromarray(corrected).save(output / "final_corrected_panorama.png")
    Image.fromarray(corrected).save(output / "final_panorama.png")
    Image.fromarray(corrected).save(
        output / "final_corrected_panorama.tiff", compression="tiff_deflate"
    )
    metadata = {
        "scene": sample["scene"],
        "capture_profile": profile_name,
        "input_resolution": sample["image_size"].int().tolist(),
        "frames": len(sample["frame_paths"]),
        "tiles_per_frame": [len(specs) for specs in sample["tile_specs"]],
        "tile_size": model_cfg["tile_size"],
        "tile_overlap": model_cfg["tile_overlap"],
        "output_resolution": [model_cfg["output_width"], model_cfg["output_height"]],
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "corrections": corrections,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(output / "final_corrected_panorama.tiff")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--config", default="../stitching/config.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "dslr_fisheye", "drone_still", "mobile", "mobile_landscape"],
    )
    args = parser.parse_args()
    run_tiled_inference(
        args.scene,
        Path(__file__).resolve().parent / args.config,
        args.output_dir,
        args.checkpoint,
        args.profile,
    )


if __name__ == "__main__":
    main()
