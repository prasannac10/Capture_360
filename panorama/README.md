# Capture360 panorama generation

This directory separates Capture360's panorama implementations:

- `pano_ai/` contains trained models, datasets, training, AI inference, and AI corrections.
- `pano_classical/` contains OpenCV and Hugin stitching/correction code.
- `stitching/` owns the shared configuration and dispatches calls to the selected engine.

## Input contract

The shared `stitching/config.yaml` selects the input profile automatically from
the frame dimensions (or set `input.profile` explicitly):

| Profile | Supported frames | Projection |
| --- | --- | --- |
| `dslr_fisheye` | 9504x6336, 4–8 images | 180-degree fisheye |
| `mobile_landscape` | 1920x1080, 20–60 images | landscape phone/pinhole |
| `drone_still` | 4096x3072, 20–30 images | pinhole/perspective |
| `mobile` | portrait images at least 3000x4000, 20–30 images | pinhole/perspective |

For AI sessions, `camera.json` must declare `fisheye_180` (or an equivalent
fisheye name) for DSLR captures, and `pinhole`/`perspective`/`phone` for drone
and mobile captures. This is validated before inference.

Each training/inference scene contains:

```text
scene_xxxxxx/
  images/        # native JPG/PNG/TIFF frames
  poses.pt       # [N,3] yaw/pitch/roll degrees
  camera.json    # projection + calibrated intrinsics/FOV
  panorama.png   # required for supervised training
```

Supported source ranges include DSLR fisheye 9504x6336, drone still 4096x3072, and mobile stills from 3000x4000 through 6120x8160. Valid frame count is 4–30; typical fisheye capture is 4–8 frames and pinhole capture is 20–30.

## Variable-resolution architecture

```text
native frames
 -> 1024x1024 overlapping tiles
 -> shared ResNet18 stride-8 encoder
 -> tile + camera + pose metadata
 -> arbitrary-N tile attention
 -> camera-aware spherical projection
 -> memory-bounded 12K x 6K decoder
 -> correction stages
 -> final panorama
```

The complete source frame is never resized to 224x224. Tiles are loaded lazily and processed in configurable chunks. The model makes two encoder passes: the first builds the global tile-attention context; the second immediately projects spatial features into the panorama canvas. This trades extra encoder compute for bounded feature memory.

## Training

From the repository root:

```bash
python -m panorama.pano_ai.train_tiled --config ../stitching/config.yaml
```

Training uses the native-resolution tiled path, supervised ground truth, validation scenes when available, mixed precision on CUDA, gradient clipping and optional EMA. The default training target is 3000x1500; production inference is 12000x6000.

## Inference

```bash
python -m panorama.pano_ai.tiled_inference --scene data/test/scene_000001 --config ../stitching/config.yaml
```

Outputs include `initial_panorama.png`, `initial_panorama.tiff`, `final_corrected_panorama.png`, `final_corrected_panorama.tiff`, and `metadata.json`.

Learned correction stages require their trained checkpoints. Baseline and single-image
advanced corrections (parallax and ghost removal) run in the tiled AI correction
pipeline. Pairwise AI stages (seam blending, overlap detection, and parallax flow)
remain disabled until their panorama-aligned reference/mask data contract and tiled
implementation are supplied. Classical lens-dot removal and sharpening remain enabled.

## Validation status

The code path has been structurally updated for variable-resolution training and inference. Actual customer-dataset GPU execution, quality benchmarking, calibration validation and peak-memory measurements are still required before treating the model as production-ready.

The legacy 224x224 ONNX export path is intentionally not advertised for this native tiled architecture; deployment export will be implemented after the Python reference contract is validated.
