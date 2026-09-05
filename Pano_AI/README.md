# Pano_AI

`Pano_AI/` contains the learned panorama stitcher. It is developed and executed independently from the Android `app/`.

## Data contract

Each scene contains `images/*.png`, `poses.pt`, and (for supervised training) one `panorama.png`. The pose representation is standardized as `[N,3]` yaw/pitch/roll **degrees** and converted once to `[N,3,3]` camera-to-world rotation matrices. There are no per-step correction labels; the supervised target is only the final panorama.

## Model

The pipeline is **Encoder -> Set Aggregator -> differentiable Spherical Fusion -> Decoder**. The encoder keeps spatial feature maps. Spherical fusion uses `grid_sample` with a 180-degree equidistant-fisheye projection and soft visibility weights.

## Android export

ONNX is the selected mobile runtime. `export.py` produces `pano_model.onnx` plus `model_metadata.json`. The fixed mobile contract is six frames at `224x224`, RGB values scaled by `1/255` with zero mean/unit std, and `[1,6,3,3]` camera-to-world rotations. Fewer frames are padded by repeating the last frame/pose; extra frames are truncated. Output is RGB `[1,3,256,512]` in `[0,1]`.

A trained checkpoint and exported ONNX artifact are not included until real training data has been supplied. The Android AI path therefore reports `AI model not available yet` when the asset is absent.
