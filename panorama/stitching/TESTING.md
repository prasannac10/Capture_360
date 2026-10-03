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
# ARCore mobile captures

For original camera JPEGs accompanied by `ar_poses.jsonl`, use the recorded
intrinsics and camera rotations for upright spherical OpenCV reprojection:

```powershell
$capture = 'C:\path\to\scene_0005\images'
python -m panorama.stitching.interface classical --images "$capture\*.jpg" --poses "$capture\ar_poses.jsonl" --output outputs/scene_0005/final_panorama.jpg
```

The CLI excludes images whose stems are absent from the pose file (including
previous panoramas). It retains all matched frames, including sky and pole
views. When all inputs share one folder, its `ar_poses.jsonl` is detected
automatically if `--poses` is omitted.

JPEG pixels must be in the original sensor orientation and resolution;
do not rotate them before this route. ARCore camera matrices use column-major
OpenGL camera-to-world conventions (+X right, +Y up, -Z forward).

The ARCore route now refines relative camera rotations using SIFT/RANSAC matches
and robust optimization constrained by the recorded poses. It estimates bounded
exposure gains on valid overlaps, selects graph-cut seams, and streams warped
frames into OpenCV multiband blending. Longitude padding handles the 360 seam.
Unmatched cameras keep their recorded poses; rejected refinement retains the
original rotations. A larger pose error can be recovered only when at least two
trusted neighboring views agree on its replacement using horizon features; the
report identifies each recovered frame. This is classical processing without
learned models.

`classical_pose.refine_poses`, `classical_pose.exposure_compensation`, and
`classical_pose.seam_blending` in `config.yaml` enable these stages independently.
The controls apply only to captures with ARCore poses. The generic feature-based
OpenCV route continues to use OpenCV's own internal stitching stages.

Translation, depth and lens calibration refinement are
not implemented by this route. Near objects and moving people can still leave
seams or distortions. Seam selection reduces averaging ghosts but is not a
guarantee of motion removal.

Optional local registration is controlled by `classical_pose.local_alignment`.
It estimates inverse optical flow at up to 1024 pixels wide against a fixed
mosaic of central source views. Forward/backward agreement, textured valid
overlap, an 8-pixel displacement bound, smoothness and grid-Jacobian checks
restrict the adjustment. A frame is retained unchanged if the smoothed field
does not reduce overlap error. This is a local appearance correction, not a
depth reconstruction. Masks and image pixels receive the same warp; longitude
wraps. Inspect geometry and coverage after enabling it, especially near poles
and moving objects. Reported overlap errors are internal diagnostics only.

`classical_pose.seam_width` controls the seam canvas (even integer, 256..4096;
API default 1024). `classical_pose.blend_bands` controls pyramid blending (1..8;
API default 5). Higher seam resolution preserves smaller boundaries but costs
more time and memory. Fewer bands narrow the blend and can expose tonal seams.
These options do not change the 4096-pixel composition cap.

`classical_scene_overrides.<scene>.enabled` activates per-scene settings.
These can override `seam_width`, `blend_bands` and `local_alignment`.
Explicit Python `pose_options` take precedence. Its `source_regions` list uses a source frame stem and a
normalized panorama polygon. Inside observed portions of each polygon, one
source owns the pixels before blending. Outside its real coverage the original
seam decision remains intact. Regions are processed in order; later regions win
if they overlap. This keeps selected structures together but can move a seam
to the polygon boundary. These are manually reviewed scene-specific settings,
not an automatic object detector. Recheck them if poses, heading or inputs change.
The dispatcher matches scene names as for object removal. Direct Python callers
can pass `source_regions` in pose options; an empty list disables the overrides.

The local-flow trial on scene 0004 reduced its internal overlap error but
introduced wavy furniture edges. It remains disabled in the checked-in config.
Furniture/ceiling structure-pinning trials also introduced steps at polygon
boundaries and are not enabled. The higher-resolution/narrower-blend trial
also exposed tonal seams at full composition size, so the original 1024-pixel
seam canvas and five bands are retained. Scene 0004's bottom 8% uses the dedicated `nadir` source wherever that
frame has valid coverage, to avoid mixing shifted floor lines at the pole.
This requires `nadir.jpg` in the capture. Other scenes retain the previous
defaults. Remaining parallax and the cloned floor patch are not solved by
these settings. Horizontal stretching at the bottom of a flat equirectangular
image is projection distortion; inspect a downward perspective view to judge
floor sharpness. Sharpening or deconvolution cannot undo this projection.

Composition is capped at 4096x2048; larger configured outputs are upscaled,
not additional native detail. Output must be 2:1. The raw output has accompanying
`_coverage.png` and `_geometry.json` files; uncovered regions remain black. The
geometry report records match errors, whether refinement was accepted, rotations,
exposure gains and coverage. Match errors are fitting diagnostics, not an
independent measure of final image quality.

The finishing pipeline keeps glare/dots/pole repair disabled unless explicitly
enabled and supplied with per-stage defect masks through `defect_masks` in its
Python API. Masks over 1% of the canvas are rejected. It does not classify bright
sky or captured poles as defects. `color` is optional CLAHE contrast enhancement;
it is separate from overlap exposure compensation. Sharpening remains optional.
The classical-only `classical_finishing.color` and `classical_finishing.sharpen`
settings are enabled by default and override the shared correction toggles for
the classical route. Set either to `false` to disable that finishing stage.
Legacy geometry toggles in `advanced_corrections` no longer operate on an already
blended panorama; metadata explains the skip and directs callers to
`classical_pose`. Source geometry belongs before composition.

Regression checks:

```powershell
python -B -m unittest panorama.pano_classical.test_corrections panorama.pano_classical.test_pose_stitcher
```

## Guided capture heading recovery

`classical_pose.recover_capture_headings` enables recovery of coherent large yaw
offsets before feature refinement. It requires the supported ARCore guided ring
layout in `capture-info.json` and at least three equator views agreeing on an
offset. This assumes the photographer followed the named capture slots; it is
not suitable for arbitrary sequences renamed to those slots. Disable the setting
if that assumption does not hold. Metadata records each changed frame. This
repairs heading errors, not translation or depth-dependent parallax.

## Removing the photographer's feet

Local alignment keeps the selected reference frame fixed and protects each
view's central region; deformation tapers into side/corner overlap regions.
Set `classical_pose.reference_frame` to an input filename stem, e.g. `mid_01`
(without `.jpg`). The default is the first input frame. Enable
`classical_pose.local_alignment` to use classical local registration. This
protection also applies to learned view refinement in spherical composition.
Spherical reprojection and global pose refinement are separate operations.


`classical_finishing.object_removal: true` activates the removal stage before
contrast and sharpening. `classical_parameters.object_removal.scenes` contains
settings keyed by scene folder name (currently `scene_0004` and `scene_0005`).
The dispatcher takes this name from the parent of the input `images` folder.
Unknown scenes skip removal and record the reason. Each entry contains
two normalized polygons covering the feet/ankles in that scene and donor
offsets pointing to clean ground. OpenCV clones that texture into the selected
regions; it does not detect feet or people automatically. Edit these regions
for a different heading or crop, or turn the stage off. Texture cloning may leave
floor-line discontinuities or tonal patches; it cannot reconstruct hidden geometry.
The configured
1.5% area limit applies only to this explicit object-removal operation.

Classical activation switches in `config.yaml`:

| Setting | Effect |
| --- | --- |
| `classical_pose.recover_capture_headings` | Repair corroborated guided-slot yaw offsets |
| `classical_pose.refine_poses` | Refine camera alignment before composition |
| `classical_pose.exposure_compensation` | Balance overlapping frame exposure |
| `classical_pose.seam_blending` | Select seams before multiband blending |
| `classical_pose.seam_width` | Resolution used to choose seams |
| `classical_pose.blend_bands` | Width of multiband transitions |
| `classical_pose.local_alignment` | Experimental guarded local registration (disabled) |
| `classical_scene_overrides.<scene>.enabled` | Preserve configured structures in individual source frames |
| `classical_finishing.object_removal` | Replace the configured object regions |
| `classical_finishing.color` | Apply final contrast enhancement |
| `classical_finishing.sharpen` | Apply final sharpening |

`advanced_corrections.toggles` are AI/legacy controls, not switches for the
current classical source-level stages. Leave them false for classical stitching.
The legacy post-blend geometry stages are skipped even if their toggles are true.

The run's metadata lists `object_removal` in `applied_stages` and reports the
repaired mask fraction. Intermediate `00_object_removal_mask.png` and
`00_object_removal.png` show exactly where and how the correction was applied.

```powershell
python -B -m unittest panorama.pano_classical.test_object_removal
```
