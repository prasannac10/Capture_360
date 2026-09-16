"""Authoritative end-to-end Python reference inference for Capture360."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from PIL import Image

from data.collate import panorama_collate_fn
from data.dataset import PanoramaDataset
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.encoder import ImageEncoder
from models.panorama_model import PanoramaModel
from pipeline import CorrectionPipeline
from utils.checkpoint import load_checkpoint
from utils.ema import EMA

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def set_deterministic(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _validate_session(session: Path) -> None:
    if not session.is_dir():
        raise FileNotFoundError(session)
    images = sorted(p for p in (session / "images").glob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No image frames found in {session / 'images'}")
    if not (session / "poses.pt").exists():
        raise FileNotFoundError(f"Missing poses.pt in {session}")
    if not (session / "camera.json").exists():
        raise FileNotFoundError(f"Missing camera.json in {session}; projection/calibration must be explicit")


def _load_cfg(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid configuration: {path}")
    return cfg


def _build_model(cfg: Dict[str, Any], device: torch.device) -> PanoramaModel:
    m = cfg["model"]
    e = m.get("encoder", {})
    model = PanoramaModel(
        ImageEncoder(m["feature_dim"], e.get("backbone", "resnet18"), bool(e.get("pretrained", True))),
        SetAggregator(m["feature_dim"]),
        PanoramaDecoder(m["feature_dim"]),
        m["pano_height"], m["pano_width"],
    ).to(device)
    model.eval()
    return model


def _save_panorama(tensor: torch.Tensor, output_stem: Path) -> None:
    image = tensor.detach().clamp(0, 1)
    if image.ndim == 4:
        image = image[0]
    array = image.permute(1, 2, 0).cpu().numpy()
    Image.fromarray((array * 255).round().astype(np.uint8)).save(output_stem.with_suffix(".png"))
    Image.fromarray((array * 65535).round().astype(np.uint16)).save(output_stem.with_suffix(".tiff"), compression="tiff_deflate")


def _load_mask(session: Path) -> Optional[np.ndarray]:
    path = session / "correction_mask.png"
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def run_reference_inference(
    session: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    device: Optional[str] = None,
    seed: int = 0,
    apply_corrections: bool = True,
) -> Dict[str, Any]:
    session, config_path, output_dir = Path(session), Path(config_path), Path(output_dir)
    _validate_session(session)
    cfg = _load_cfg(config_path)
    set_deterministic(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    dataset = PanoramaDataset(session.parent, has_gt=False)
    matches = [i for i, scene in enumerate(dataset.scenes) if scene.resolve() == session.resolve()]
    if not matches:
        raise ValueError(f"Session is not a dataset scene: {session}")
    batch = panorama_collate_fn([dataset[matches[0]]])

    model = _build_model(cfg, selected_device)
    checkpoint = cfg.get("inference", {}).get("checkpoint")
    if not checkpoint:
        raise ValueError("inference.checkpoint must be configured")
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = config_path.parent / checkpoint_path
    ema = EMA(model, cfg["training"]["ema_decay"]) if cfg["training"].get("use_ema", False) else None
    load_checkpoint(str(checkpoint_path), model, device=selected_device, ema=ema, use_ema=ema is not None)
    model.eval()

    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        initial = model(
            batch["images"].to(selected_device),
            batch["rotations"].to(selected_device),
            batch["mask"].to(selected_device),
            batch["camera_params"].to(selected_device),
        ).clamp(0, 1)

    _save_panorama(initial, output_dir / "initial_panorama")
    final = initial
    correction_meta: Dict[str, Any] = {"enabled": False, "stages": []}

    if apply_corrections:
        correction_cfg = cfg.get("correction", {})
        advanced_cfg = cfg.get("advanced_corrections", {})
        toggles = dict(correction_cfg.get("toggles", {}))
        toggles.update(advanced_cfg.get("toggles", {}))
        checkpoints = dict(correction_cfg.get("checkpoints", {}))
        checkpoints.update(advanced_cfg.get("checkpoints", {}))
        correction = CorrectionPipeline(
            toggles=toggles,
            checkpoints=checkpoints,
            device=selected_device,
            allow_untrained=bool(correction_cfg.get("allow_untrained", False)),
        )
        final_np = correction.run(
            (initial[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8),
            correction_mask=_load_mask(session),
        )
        final = torch.from_numpy(final_np).permute(2, 0, 1).float().div(255).unsqueeze(0)
        correction_meta = {"enabled": True, "stages": [k for k, v in toggles.items() if v]}

    _save_panorama(final, output_dir / "final_panorama")
    projection = "pinhole" if bool(batch["camera_params"][0, 0, 4].item() > 0.5) else "fisheye_180"
    metadata = {
        "schema_version": 1,
        "scene": session.name,
        "num_frames": int(batch["mask"].sum().item()),
        "projection": projection,
        "camera_params": batch["camera_params"][0, 0].cpu().tolist(),
        "panorama_size": [int(final.shape[-2]), int(final.shape[-1])],
        "model_checkpoint": str(checkpoint_path),
        "device": str(selected_device),
        "seed": seed,
        "deterministic": True,
        "corrections": correction_meta,
        "outputs": ["initial_panorama.png", "initial_panorama.tiff", "final_panorama.png", "final_panorama.tiff", "metadata.json"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture360 Python reference inference")
    parser.add_argument("--session", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-corrections", action="store_true")
    args = parser.parse_args()
    run_reference_inference(args.session, args.config, args.output, args.device, args.seed, not args.no_corrections)


if __name__ == "__main__":
    main()
