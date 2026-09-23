"""Command-line interface for testing Capture360 panorama stitching.

Run from the repository root:
    python -m panorama.stitching.interface classical --images samples/*.jpg --output outputs/classical
    python -m panorama.stitching.interface ai --session data/test/scene_000001 --output outputs/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from glob import glob
from .dispatcher import stitch


def _classical(args: argparse.Namespace) -> None:
    image_paths: list[str] = []
    for pattern in args.images:
        matches = glob(pattern)
        image_paths.extend(matches or [pattern])

    if not image_paths:
        raise ValueError("Provide one or more paths with --images")
    result = stitch(
        image_paths,
        args.output,
        config_path=args.config,
        profile=args.profile,
        engine="classical",
    )
    print(f"Classical panorama output directory: {Path(args.output).resolve()}")
    print(f"Output shape: {result.shape[1]}x{result.shape[0]}")


def _ai(args: argparse.Namespace) -> None:
    output = stitch(
        Path(args.session),
        args.output,
        config_path=args.config,
        profile=args.profile,
        engine="ai",
        checkpoint=args.checkpoint,
    )
    print(f"AI panorama output directory: {Path(output).resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test Capture360 panorama stitching")
    parser.add_argument(
        "--config",
        default="panorama/stitching/config.yaml",
        help="Shared panorama YAML configuration.",
    )
    subparsers = parser.add_subparsers(dest="engine", required=True)

    classical = subparsers.add_parser(
        "classical", help="Stitch image files with OpenCV."
    )
    classical.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Input image paths; shell globs are supported.",
    )
    classical.add_argument(
        "--output",
        required=True,
        help="Output directory, for example outputs/classical.",
    )
    classical.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "dslr_fisheye", "drone_still", "mobile", "mobile_square", "mobile_landscape"],
    )
    classical.set_defaults(func=_classical)

    ai = subparsers.add_parser(
        "ai", help="Stitch a Capture360 session with the tiled AI model."
    )
    ai.add_argument(
        "--session",
        required=True,
        help="Session directory containing images/, poses.pt, and camera.json.",
    )
    ai.add_argument(
        "--output",
        required=True,
        help="Parent output directory; a scene-named folder is created inside it.",
    )
    ai.add_argument(
        "--checkpoint",
        help="Optional tiled-model checkpoint path; otherwise inference.checkpoint is used.",
    )
    ai.add_argument(
        "--profile",
        default="auto",
        choices=["auto", "dslr_fisheye", "drone_still", "mobile", "mobile_landscape"],
    )
    ai.set_defaults(func=_ai)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
