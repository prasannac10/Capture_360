# Stitching test interface

Run commands from the repository root, `C:\AI_Projects\Capture_360`.

## Classical OpenCV stitching

```powershell
python -m panorama.stitching.interface classical `
  --images "C:\images\fisheye\*.jpg" `
  --output "C:\outputs\classical_panorama"
```

Inputs must be image paths of a single capture session, with all frames sharing
the same dimensions. The profile is selected automatically:

| Capture type | Frames | Dimensions | Projection |
| --- | ---: | --- | --- |
| DSLR fisheye | 4–8 | approximately 9504x6336 | Fisheye |
| Drone still | 20–30 | approximately 4096x3072 | Perspective |
| Mobile | 20–30 | portrait, at least 3000x4000 | Perspective |

Pass `--profile dslr_fisheye`, `--profile drone_still`, or `--profile mobile`
to require a particular profile instead of auto-detection.

The output folder contains `initial_panorama.png`, every enabled correction
stage (`01_glare.png`, etc.), `final_panorama.png`, and `metadata.json`.

## AI tiled stitching

```powershell
python -m panorama.stitching.interface ai `
  --session "C:\data\test\scene_000001" `
  --output "C:\outputs" `
  --checkpoint "C:\AI_Projects\Capture_360\panorama\pano_ai\checkpoints\panorama_tiled_best.pt"
```

The session must have this structure:

```text
scene_000001/
  images/              # JPG, PNG, WebP, TIFF; one capture session
  poses.pt             # Torch tensor [number_of_frames, 3] of yaw/pitch/roll degrees
  camera.json          # calibrated camera metadata and projection
```

`camera.json` must be compatible with the detected profile:

```json
// DSLR fisheye
{"projection": "fisheye_180", "fov_deg": 180.0}

// Drone/mobile pinhole
{"projection": "pinhole", "image_width": 4096, "image_height": 3072,
 "horizontal_fov_deg": 84.0}
```

The command creates `outputs/scene_000001/` containing the initial panorama,
every enabled correction stage, the final panorama, and `metadata.json`. It requires a tiled model
checkpoint. If `--checkpoint` is omitted, it uses `inference.checkpoint` from
`panorama/stitching/config.yaml`.

## Python API

```python
from pathlib import Path
from panorama.stitching import stitch

# Classical: iterable of image paths -> one panorama image.
stitch(
    [Path("C:/images/01.jpg"), Path("C:/images/02.jpg")],
    "C:/outputs/classical.jpg",
    engine="classical",
)

# AI: session directory -> scene output directory.
stitch(
    "C:/data/test/scene_000001",
    "C:/outputs",
    engine="ai",
    checkpoint="C:/checkpoints/panorama_tiled_best.pt",
)
```
