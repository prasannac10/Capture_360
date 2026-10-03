"""Capture-profile detection and validation shared by both stitchers."""

from __future__ import annotations

from typing import Any


def select_profile(
    width: int, height: int, input_config: dict[str, Any], requested: str = "auto"
) -> tuple[str, dict[str, Any]]:
    """Return the requested or resolution-matched capture profile."""
    profiles = input_config["profiles"]
    if requested != "auto":
        if requested not in profiles:
            raise ValueError(
                f"Unknown input profile {requested!r}; choose one of {sorted(profiles)}"
            )
        profile = profiles[requested]
        _validate_dimensions(width, height, requested, profile)
        return requested, profile
    # Prefer established exact profiles before overlapping resolution ranges.
    for name, profile in profiles.items():
        if _matches(width, height, name, {k: v for k, v in profile.items() if k != 'resolution_range'}):
            return name, profile
    for name, profile in profiles.items():
        if _matches(width, height, name, profile):
            return name, profile
    supported = ", ".join(
        f"{name} ({p['width']}x{p['height']})" for name, p in profiles.items()
    )
    raise ValueError(
        f"Unsupported input resolution {width}x{height}; supported profiles: {supported}"
    )


def _matches(width: int, height: int, name: str, profile: dict[str, Any]) -> bool:
    limits = profile.get('resolution_range')
    if limits and width > 0 and height > 0:
        short, long = sorted((width, height))
        if (limits['min_short_edge'] <= short <= limits['max_short_edge']
                and long <= limits['max_long_edge']):
            return True
    if name == "dslr_fisheye":
        # DSLR bodies may store the same sensor orientation as portrait or
        # landscape depending on EXIF/orientation handling.  Fisheye capture
        # geometry is unchanged, so accept either dimension ordering.
        expected_width, expected_height = profile["width"], profile["height"]
        return any(
            abs(actual_width - target_width) <= target_width * 0.05
            and abs(actual_height - target_height) <= target_height * 0.05
            for actual_width, actual_height, target_width, target_height in (
                (width, height, expected_width, expected_height),
                (width, height, expected_height, expected_width),
            )
        )
    if name == "mobile":
        return (
            width >= profile["width"] and height >= profile["height"] and height > width
        )
    if name == "mobile_square":
        return (
            abs(width - profile["width"]) <= profile["width"] * 0.05
            and abs(height - profile["height"]) <= profile["height"] * 0.05
        )
    return (
        abs(width - profile["width"]) <= profile["width"] * 0.05
        and abs(height - profile["height"]) <= profile["height"] * 0.05
    )


def _validate_dimensions(
    width: int, height: int, name: str, profile: dict[str, Any]
) -> None:
    if not _matches(width, height, name, profile):
        if profile.get('resolution_range'):
            limits = profile['resolution_range']
            raise ValueError(
                f"{name} accepts short edge {limits['min_short_edge']}..{limits['max_short_edge']} "
                f"and long edge <= {limits['max_long_edge']}, or its original profile size; got {width}x{height}"
            )
        qualifier = "at least" if name == "mobile" else "approximately (either orientation for dslr_fisheye)"
        raise ValueError(
            f"{name} expects {qualifier} {profile['width']}x{profile['height']}; got {width}x{height}"
        )


def validate_frame_set(
    dimensions: list[tuple[int, int]],
    input_config: dict[str, Any],
    requested: str = "auto",
) -> tuple[str, dict[str, Any]]:
    if not dimensions:
        raise ValueError("At least one input image is required")
    name, profile = select_profile(*dimensions[0], input_config, requested)
    for width, height in dimensions[1:]:
        _validate_dimensions(width, height, name, profile)
    return name, profile
