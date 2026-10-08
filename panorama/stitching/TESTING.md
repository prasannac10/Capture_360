# Stitching test interface

Run commands from the repository root, `C:\AI_Projects\Capture_360`.

## Classical OpenCV stitching

```powershell
python -m panorama.stitching.interface classical `
  --images "C:\images\fisheye\*.jpg" `
  --output "C:\outputs\classical_panorama"
```

Inputs must belong to one capture session and match the selected profile.
For calibrated captures, each image must match its own recorded dimensions and
intrinsics. Exact profiles take priority over overlapping resolution ranges in
auto mode; dimensions alone cannot distinguish a drone from a mobile camera.

| Capture type | Frames | Dimensions | Projection |
| --- | ---: | --- | --- |
| DSLR fisheye | 4–8 | approximately 9504x6336 | Fisheye |
| Drone still | 20–30 | approximately 4096x3072 | Perspective |
| Mobile | 20–30 | portrait, at least 3000x4000 | Perspective |

Both `drone_still` and `mobile` additionally accept positive dimensions with a
long edge <=3500 and short edge <=2100, in either orientation. Their original
profile sizes remain accepted. The range is a compatibility check, not a minimum
quality recommendation. Other exact mobile profiles retain their configured
frame counts; inspect `input.profiles` for the full list.

Pass `--profile dslr_fisheye`, `--profile drone_still`, or `--profile mobile`
to require a particular profile instead of auto-detection.

The output folder contains `initial_panorama.png`, every enabled correction
stage (`01_glare.png`, etc.), `final_panorama.png`, and `metadata.json`.

## AI tiled stitching

```powershell
python -m panorama.stitching.interface ai `
  --session "C:\data\test\scene_000001" `
  --output "C:\outputs" `
  --profile drone_still `
  --checkpoint "C:\AI_Projects\Capture_360\panorama\pano_ai\checkpoints\panorama_native_detail_best.pt"
```

The default `tiled_neural` / `rgb_residual` architecture requires a newly trained
`panorama_native_rgb_residual_v7` checkpoint. It samples original RGB at native
12K output coordinates and learns bounded residual corrections; no classical
stitcher is used. Existing feature-only v4 checkpoints do not contain this head.

A canonical session can contain `capture.json` plus its referenced source images
(see [camera contract](../pano_ai/README.md#shared-camera-input)). The legacy
pose/camera adapter accepts:

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
{"projection": "pinhole", "image_width": 3500, "image_height": 2100,
 "horizontal_fov_deg": 84.0}
```

The command creates `outputs/scene_000001/` containing the initial panorama,
every enabled correction stage, the final panorama, and `metadata.json`. It requires a tiled model
checkpoint. If `--checkpoint` is omitted, it uses `inference.checkpoint` from
`panorama/stitching/config.yaml`.

Default output size is 12000x6000, controlled by `model.output_width/height`.
The `stitching.output_*` settings control classical dispatch instead.
In neural mode the scene folder contains
`initial_panorama.png`, `initial_panorama.tiff`, `final_panorama.png`,
`final_corrected_panorama.png`, `final_corrected_panorama.tiff`, `metadata.json`,
and any enabled correction intermediates/defect masks. TIFFs are lossless 8-bit
RGB, not HDR. Metadata records `detail_mode`, checkpoint/inference contracts,
selected model/EMA weights, source/output sizes and exposure diagnostics.
Correction toggles are disabled until separately trained weights are validated.

For a v4 comparison, use a separate config with `model.detail.mode: features`
and the matching checkpoint. Native `residual_scale` and exposure settings must
match the v5 training config. `allow_legacy_checkpoint` cannot override a detail
architecture or native residual-scale mismatch. The alternative
`ai_pipeline.mode: source_preserving` uses the geometric compositor and optional
separate view heads; it does not accept `--checkpoint` for a panorama decoder.

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
    profile="drone_still",
    checkpoint="C:/checkpoints/panorama_native_detail_best.pt",
)
```

## Neural native-detail validation

Run from the repository root with torch/torchvision and the training requirements
installed. These tests create small synthetic temporary datasets and do not
require real captures or pretrained downloads:

```powershell
python -B -m unittest panorama.pano_ai.tests.test_native_detail
python -B -m unittest discover -s panorama/pano_ai/tests
```

The native-detail tests cover source-texture identity before learning, winner
ownership of conflicting views, overlap disagreement, periodic region projection,
learned tile/crop continuity, encoder/head gradients with activation checkpointing,
target-size rejection, actual training/EMA checkpoint reload, neural inference,
and architecture-mismatch rejection. A crop test samples the actual 12K
coordinate lattice; the suite is not full-size GPU validation or a quality study.
The broader `train.smoke_training` workflow deliberately selects the legacy
feature-only decoder for its small synthetic references.

For real training, each source scene needs valid calibration and an aligned,
genuine 12000x6000 `panorama.png`. Split by physical scene. The native trainer
rejects smaller targets and renders 1024-pixel crops at the full output scale;
it does not resize the target to 3K. Four crops per scene per epoch are the
default, with a 50% preference for source overlap disagreement. Validation uses
up to five fixed corner/centre crops, so logged metrics are crop averages.
Detailed settings and data layouts are in the [training guide](../pano_ai/README.md#native-detail-configuration-and-checkpoint-migration).

Train with the prepared two-stage bundle or use the standalone panorama trainer:

```powershell
python -m panorama.pano_ai.train.run_training --config panorama/pano_ai/train/configs/stitching_correction_job.yaml
# Alternative: panorama only; configure its training_data/val_data first.
python -m panorama.pano_ai.train_tiled --config ../stitching/config.yaml
```

The standalone trainer and `tiled_inference` CLI resolve `--config` relative to
`panorama/pano_ai/`, while
the stitching interface resolves it from the working directory. Configured
checkpoint paths resolve relative to the shared YAML; explicit `--checkpoint`
paths resolve from the working directory. Training writes
`panorama_native_detail_best.pt` and `panorama_native_detail_last.pt`; panorama
and combined correction have separate checkpoints/supervision.

After retraining, run a held-out scene at 12K on the intended GPU. Confirm output
dimensions, metadata detail mode/contract, memory and latency. Compare the v5
result with an aligned reviewed reference and an explicit v4 baseline at identical
dimensions/heading. Inspect straight edges, fine texture, overlapping objects,
exposure transitions, poles and the actual longitude join. Record metrics/crops:

```powershell
python -m panorama.stitching.visual_regression --candidate outputs/native/scene_001/final_panorama.png --reference data/references/scene_001/panorama.png --out outputs/native_visual_check
```

Paths above are placeholders for a prepared held-out scene. Metrics are RGB MAE,
edge error, high-pass texture error and longitude-join error; outputs include
native centre/bottom/join side-by-side crops. Use `--limits limits.json` with
thresholds derived from held-out captures to fail regressions. Supply `--original`
and `--mask` only for a masked correction preservation check, not for the global
neural stitcher. Differences from an unreviewed baseline are not proof of quality.
Rotation-only geometry still leaves translation parallax unresolved; hard
ownership can reveal seams, and the residual head is not an optical-flow model.

## T4 versus H100 training benchmark

Run the same code revision, config and representative training scene on each
instance. The scene must include calibrated sources and `panorama.png`. Do not
change tile batch/crop sizes for the first comparison; that changes the workload.
From the repository root on T4:

```bash
python -m panorama.pano_ai.train.benchmark --scene /data/panorama/train/scene_001 --config panorama/stitching/config.yaml --out outputs/benchmark_t4 --warmup 2 --steps 5
```

Repeat on H100 with identical arguments except `--out outputs/benchmark_h100`.
Paths are examples; use the same copied scene on both hosts. Optional
`--checkpoint /path/to/panorama_native_detail_best.pt` initializes both from the
same matching model weights. Otherwise both runs use the configured encoder
initialization and seed. Use empty/new output directories. This performs real
optimizer updates on an in-memory model without overwriting training checkpoints.
CUDA is required; `--allow-cpu` exists only for synthetic execution tests.

`report.json` includes synchronized wall time per optimizer step, peak PyTorch
allocated/reserved VRAM, scene/code/initial-weight fingerprints, effective config,
software versions, metrics and one-second `nvidia-smi` utilization by physical
GPU UUID. Scene/reference/calibration/exposure loading is timed separately as
setup. Each step is one training crop, not an entire scene or epoch.
The timings include the production crop sampler, two encoder passes, loss,
backward, optimizer, optional EMA and training metrics.

A separate instrumented step writes `trace.json` and named CPU/device stage
timings for source-tile loading, native-photo loading, native RGB sampling and
projection, feature projection, encoder, decoder, forward, loss, backward and
optimizer. Load the Chrome trace in a compatible trace viewer to inspect CPU
gaps, GPU kernels and nested work. Inclusive stage times overlap and cannot be
summed; encoder ranges also appear during checkpoint recomputation in backward.
The profiler step is excluded from reported throughput and measured peak VRAM.
GPU utilization is a sampled observation, not exact kernel occupancy; if telemetry
is unavailable, the report records an error instead of fabricated utilization.

Bring the two report files onto one machine and compare:

```bash
python -m panorama.pano_ai.train.compare_benchmarks --first outputs/benchmark_t4/report.json --second outputs/benchmark_h100/report.json --out outputs/t4_h100_comparison.json
```

The comparison rejects scene/code/initial-weight or relevant config mismatches
and reports the measured step speedup and both stage breakdowns. Match software
and data storage as well as GPU settings; a different CPU/storage configuration
can change loading time. This benchmark excludes epoch validation/checkpoint
saving, so its speedup must not be presented as a guaranteed full-run speedup.
No T4/H100 timings are established until these commands run on accessible GPUs.

Check the benchmark implementation with:

```powershell
python -B -m unittest panorama.pano_ai.tests.test_benchmark
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

### Coverage-aware native detail training

Native RGB training supervises observed pixels only. L1 is normalized by covered
pixels; edge loss requires both adjacent pixels, and SSIM requires a fully covered
window. Perceptual inputs use identical zero context outside coverage. Validation
metrics exclude uncovered pixels and entirely uncovered validation crops are skipped.
Use an unedited `panorama.png` or `Stitched.jpg` at the configured output size;
`Edited.jpg` is never selected automatically. Labels or retouching within observed
regions still require a clean reference. Calibration and decoded RGB photos remain
required; a raw DNG/PTGui folder is not directly accepted.

Native inference writes `coverage_mask.png` (white = observed), and automatically
provides missing coverage to the configured `nadir_zenith` correction model. With
that correction disabled, unsupported areas remain black. Filling these areas
requires an appropriate trained correction checkpoint or additional capture coverage;
the stitching decoder does not invent missing sky. Native exposure gains now use
the same linear-light transfer function as encoder preprocessing. Retrain the native
model after this preprocessing change; previous non-unity gain runs are not comparable.

### Full-resolution RAW preparation

Install `rawpy` in the preparation environment, then run:

```powershell
python -m panorama.pano_ai.data.prepare_native --scene "C:/data/raw/scene_0001" --output "C:/data/prepared/scene_0001"
```

This new command decodes DNGs with `half_size=False`, no automatic orientation
rotation, camera white balance and no automatic brightness. It saves full-size
full-resolution 8-bit sRGB JPEGs (quality 100, 4:4:4; JPEG compression remains lossy), copies the original 12000?6000 `Stitched.jpg` without
resizing, retains `Panorama.pts`, and never deletes source DNGs. Output must be a
new directory. `Edited.jpg` is not used. A failed preparation may leave partial
output; use a new directory on retry.

Supply `--capture /path/to/capture.json` only when its calibration already matches
the full-size, unrotated decoded RGB pixels. Half-size calibration is rejected.
Without calibration, `preparation.json` explicitly marks the result as requiring
calibration before training. This command does not convert PTGui lens distortion
or poses; verified calibration/undistortion remains required. RAW decoding has
not been validated on the supplied real scene locally because rawpy is absent.
The default drone profile permits at most 30 frames; the supplied 33-frame scene
needs an explicitly adjusted frame limit after calibration. Train with the default
`rgb_residual` mode and a new v7 checkpoint.

Existing full-resolution `.jpg`/`.jpeg` frames in `images/` are accepted by
`prepare_native` and copied byte-for-byte, with no resize or recompression.
A JPEG-only scene needs neither DNG files nor rawpy. Calibration must match
the JPEG pixel dimensions and orientation. Duplicate JPEG/DNG stems are rejected;
select one source version per frame.

## Seam-focused preparation and next training run

Keep the configured 12000?6000 output and genuine target dimensions, and retain
`rgb_residual`. The v7 photometric base is unchanged. New training samples actual
source ownership boundaries rather than general disagreement; a four-pixel band
around adjacent observed source changes defines the seam mask. Missing coverage
is excluded. `loss.supervised.seam_weight: 0.2` adds boundary-band L1 and reference
gradient error to existing losses. Training logs `seam_l1`, `seam_gradient_l1` and
`seam_pixels`; zero seam error with zero seam pixels means no seam was evaluated.
Validation uses five fixed crops plus up to three deterministic boundary crops.
Crop metrics average by crop, so compare runs with the same evaluation regions.
These changes provide targeted supervision, not a learned flow/alignment module.

Compare native ownership, the existing photometric base, and trained inference
before correction on identical native-scale crops:

```powershell
python -m panorama.pano_ai.train.diagnose_seams --scene /data/panorama/val/scene_0001 --config panorama/stitching/config.yaml --out outputs/seams_scene_0001 --region 1500 2000 1024 1024 --prediction /data/inference/scene_0001/initial_panorama.png
```

Region arguments are top, left, height, width in the full output pixel grid.
Choose sky and wire/building boundary crops. The tool saves ownership, photometric,
trained and reference crops, coverage/seam masks and JSON metrics. Omit
`--prediction` to compare the two source baselines only. It rejects resized
references/predictions and existing output directories. Geometry must be verified
from calibrated source projections; broken edges may still require alignment.

For correction training, create a local manifest with version 1 and pairs using
actual initial AI outputs as `before`, clean reviewed aligned panoramas as `after`,
unique `scene_id`, `site`, `split` (train/val), `domain`, `projection: equirectangular`,
`alignment_verified: true`, `before_source: ai_stitcher`, and `stitcher_checkpoint`
(the checkpoint path/version). Do not use edited labels as stitching references.
Both image paths must resolve locally relative to the manifest. The downloaded
S3 manifest must first be materialized and reviewed; its object keys are not local
paths and cannot be relabelled as AI outputs.

```powershell
python -m panorama.pano_ai.data.prepare_pairs --manifest /data/ai_pairs.json --out outputs/ai_output_correction_bundle --require-ai-outputs
python -m panorama.pano_ai.train.run_training --config panorama/pano_ai/train/configs/ai_output_correction_job.yaml
```

Packaging preserves checkpoint provenance and prevents site leakage. Keep combined correction
disabled during stitcher evaluation. Enable the chosen correction checkpoint only
after comparing held-out initial versus corrected outputs. Missing sky still
requires separately validated filling or additional capture coverage.
