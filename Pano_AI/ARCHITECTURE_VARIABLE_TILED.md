# Capture360 Variable-Resolution Panorama Architecture

## Objective

Generate a production 12,000 x 6,000 equirectangular panorama while preserving the native detail of:

- DSLR fisheye: 9504 x 6336
- Drone still: 4096 x 3072
- Mobile still: 3000 x 4000 through 6120 x 8160

The model must not resize an entire source image to 224 x 224 before inference.

## Architecture

```text
Variable-resolution source frames
          |
          v
Original-resolution 1024x1024 tiles + 128 overlap
          |
          v
Shared ResNet18 tile encoder (stride 8)
          |
          +---- visual feature maps
          |
          +---- tile tokens
          |
          +---- tile x/y + size
          |
          +---- camera intrinsics/FOV/projection
          |
          +---- frame pose
          v
Tile metadata fusion
          |
          v
Multi-head attention over arbitrary N x T tiles
          |
          +---- per-tile contextual features
          |
          +---- global scene token
          v
Camera-aware spherical projection
          |
          v
Multi-scale panorama decoder
          |
          v
12,000 x 6,000 RGB panorama
          |
          v
Memory-bounded high-resolution correction pipeline
          |
          v
Final 12K x 6K PNG/TIFF
```

## Why tiles

A 12K x 6K RGB panorama contains 216 million output values. A full-resolution fully-connected network would be impractical. Convolutional feature maps and overlapping tiles preserve local detail without requiring the whole source to exist in every GPU activation.

## Variable input count

The model accepts arbitrary valid frame counts (4-8 fisheye and 20-30 pinhole are the recommended ranges). Each frame can produce a different number of tiles. Padding/masks are used only for batching; invalid tiles never participate in attention or spherical projection.

## Geometry

Each tile carries its original pixel origin and dimensions. Camera metadata carries projection type, intrinsics and FOV. Per-frame yaw/pitch/roll is converted to a rotation matrix. The spherical projector maps equirectangular rays back into each source image and then into the correct tile coordinate system.

## Resolution strategy

The tile encoder operates on 1024 x 1024 native pixels and retains stride-8 feature maps (128 x 128). The spherical feature canvas is intentionally smaller than the final panorama to keep training/inference tractable. The decoder reconstructs the requested production output size. Training uses a configurable smaller target (default 3000 x 1500) and inference uses the production 12000 x 6000 target.

## Correction stages

Corrections remain independent models. Baseline learned stages (glare, nadir/zenith, color) and advanced learned stages (parallax, ghost removal) are applied using overlapping 1024 tiles so a 12K image does not need to pass through a correction U-Net as one giant activation. Lens-dot removal and sharpening remain classical finishing stages. Pairwise advanced stages such as seam blending, overlap detection and parallax-flow require explicit auxiliary reference/mask data and remain separately enabled after their checkpoints and contracts are validated.

## Production principle

The 12K output size is a tensor/output-resolution requirement, not a requirement for 216 million fully connected neurons. Spatial convolution and tiled decoding are used to make the output feasible.
