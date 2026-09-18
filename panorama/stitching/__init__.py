"""Configuration-driven panorama stitching entry points."""

from .dispatcher import load_config, stitch

__all__ = ["load_config", "stitch"]
