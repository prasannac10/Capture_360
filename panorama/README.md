# Capture360 panorama generation

Capture360 provides classical panorama stitching and a neural path with native
RGB detail refinement followed by optional learned correction. The neural model
requires new trained weights and held-out drone/mobile quality validation.

## Project layout

| Directory | Purpose |
| --- | --- |
| `pano_classical/` | OpenCV/Hugin stitching, calibrated spherical composition and classical corrections |
| `pano_ai/` | AI models, data preparation, training, inference and learned corrections |
| `stitching/` | Shared configuration, camera-data contract and engine dispatch |

## Pipeline

The default AI mode is `tiled_neural` with `model.detail.mode: rgb_residual`.
Two streamed encoder passes build scene features; a native RGB skip samples
original photos directly at 12000x6000 coordinates. Single-source ownership
avoids broad overlap averaging. A learned detail head uses RGB, coverage,
disagreement and scene features to predict bounded residual corrections.
Training uses genuine 12K references and native-scale output crops with edge loss.

The alternative `source_preserving` mode uses geometric projection, graph-cut
seams and multiband blending, with optional separately trained view refinement.
Training combined correction does not train either the neural panorama model or
the optional alignment/blending heads. Post-stitch correction is disabled by default.

Input profiles and stage toggles live in [stitching/config.yaml](stitching/config.yaml).
Calibrated paths share per-frame intrinsics, rotations, dimensions and filenames
through `capture.json` or supported pose adapters. Current composition uses
rotation-only geometry; translation and depth metadata do not imply depth fusion.

## Training and documentation

Use the [AI data preparation and training guide](pano_ai/README.md) for dependency
installation, commands, synthetic smoke tests, mobile validation and AWS/S3 setup.
The two-stage job trains `panorama` and `combined` independently. Stitching needs
calibrated source frames plus aligned, genuine 12000x6000 `panorama.png` references;
correction needs reviewed before/after pairs. Correction-only `combined` training
does not require per-stage labels or defect masks. Inference needs calibration
and compatible checkpoints, but no reference panorama.

See the [architecture block diagrams](ARCHITECTURE_VARIABLE_TILED.md) for training,
inference, neural tensor shapes, the native residual head and the correction U-Net.
See [testing instructions](stitching/TESTING.md#neural-native-detail-validation)
for checkpoint migration, regression commands and held-out quality checks.

## Validation status

Synthetic smoke tests check execution and data contracts. Production activation
still requires held-out real mobile evaluation, full-panorama visual review and
GPU resource checks. Combined correction can change any pixel and must be enabled
alone among post-blend correction stages.

The neural detail model uses `panorama_native_rgb_residual_v7` checkpoints;
feature-only v4 checkpoints cannot load its new head. Native RGB sampling avoids
the alternative geometric compositor's 4096-pixel working-width cap. Neither
12K export nor a native RGB skip guarantees recovered detail, corrected
translation parallax or reconstruction of missing surfaces.
