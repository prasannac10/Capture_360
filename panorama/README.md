# Capture360 panorama generation

Capture360 provides classical panorama stitching and an AI path that uses
geometric stitching with optional learned correction. Start with the geometric
baseline; enable learned models after validating their checkpoints on real
mobile captures.

## Project layout

| Directory | Purpose |
| --- | --- |
| `pano_classical/` | OpenCV/Hugin stitching, calibrated spherical composition and classical corrections |
| `pano_ai/` | AI models, data preparation, training, inference and learned corrections |
| `stitching/` | Shared configuration, camera-data contract and engine dispatch |

## Pipeline

The recommended AI mode, `source_preserving`, projects calibrated source images,
compensates exposure and blends seams. Optional learned view refinement retains
the geometric result when proposals fail reliability checks. A trained combined
correction model can then process the panorama in overlapping tiles.

The `tiled_neural` panorama decoder is a separate experimental route. Training
the combined correction model does not train that decoder or the optional
alignment and blending heads.

Input profiles and stage toggles live in [stitching/config.yaml](stitching/config.yaml).
Calibrated paths share per-frame intrinsics, rotations, dimensions and filenames
through `capture.json` or supported pose adapters. Current composition uses
rotation-only geometry; translation and depth metadata do not imply depth fusion.

## Training and documentation

Use the [AI data preparation and training guide](pano_ai/README.md) for dependency
installation, commands, synthetic smoke tests, mobile validation and AWS/S3 setup.
The recommended first training task is `combined`, using reviewed, aligned
PTGui-before / corrected-after panoramas. This task does not require separate
per-stage labels or defect masks. Calibrated mobile inference needs source frames
and camera metadata, but no corrected reference panorama.

See the [architecture block diagrams](ARCHITECTURE_VARIABLE_TILED.md) for training,
inference, the correction U-Net and the experimental decoder.

## Validation status

Synthetic smoke tests check execution and data contracts. Production activation
still requires held-out real mobile evaluation, full-panorama visual review and
GPU resource checks. Combined correction can change any pixel and must be enabled
alone among post-blend correction stages.

The geometric compositor currently works at at most 4096 pixels wide and enlarges
larger requested outputs. A 12000x6000 output therefore does not establish native
12K detail or recovery of missing surfaces.
