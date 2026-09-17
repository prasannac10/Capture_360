# Pano_AI — variable-resolution panorama pipeline

Pano_AI is the Python reference implementation for Capture360's native-resolution panorama generation.

## Input contract

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

From `Pano_AI`:

```bash
python train_tiled.py --config config.yaml
```

Training uses the native-resolution tiled path, supervised ground truth, validation scenes when available, mixed precision on CUDA, gradient clipping and optional EMA. The default training target is 3000x1500; production inference is 12000x6000.

## Inference

```bash
python tiled_inference.py --scene ../data/test/scene_000001 --config config.yaml
```

Outputs include `initial_panorama.png`, `initial_panorama.tiff`, `final_corrected_panorama.png`, `final_corrected_panorama.tiff`, and `metadata.json`.

Correction models are disabled by default until their trained checkpoints are present. Classical lens-dot removal and sharpening remain enabled. Enable a learned correction only after its checkpoint and validation are available.

## Validation status

The code path has been structurally updated for variable-resolution training and inference. Actual customer-dataset GPU execution, quality benchmarking, calibration validation and peak-memory measurements are still required before treating the model as production-ready.

The legacy 224x224 ONNX export path is intentionally not advertised for this native tiled architecture; deployment export will be implemented after the Python reference contract is validated.
