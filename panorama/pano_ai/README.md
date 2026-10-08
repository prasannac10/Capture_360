# Panorama AI: data preparation and training

This is the single guide for preparing data and starting training. Follow steps
1-5 first; mobile validation and generated examples are optional next steps.

- [Prepare your pairs](#1-list-three-real-scene-pairs-for-the-first-trial)
- [Start training](#5-start-training)
- [Mobile validation](#6-add-real-mobile-validation-when-available)
- [Generate mobile-style examples](#7-optional-automatically-generate-mobile-style-examples)
- [AWS and S3](#aws-and-s3)
- [Architecture block diagrams](../ARCHITECTURE_VARIABLE_TILED.md)
- [Native-detail configuration and checkpoint migration](#native-detail-configuration-and-checkpoint-migration)
- [Neural regression tests](#regression-evaluation)
- [T4/H100 timing and GPU profiling](../stitching/TESTING.md#t4-versus-h100-training-benchmark)

**The overall pipeline is AI stitching first, then panorama correction.**
Use the two-stage training workflow below for both models. Steps 1-7 describe
correction-only training from PTGui-before / corrected-after pairs.
The optional mobile-example generator is described in step 7. No separate glare,
ghost, alignment or blending labels are needed for this first workflow.

All commands below run in PowerShell from the repository root:

```powershell
Set-Location C:\AI_Projects\Capture_360
$trainPython = '.\.venv-ai-tests\Scripts\python.exe'
```

This existing environment was tested on CPU. On another machine, create/install
a suitable environment and set `$trainPython` to its Python executable. Install
dependencies with `& $trainPython -m pip install -r panorama/requirements-training.txt`.
The training requirements include the shared `../requirements.txt`; optional
metrics/export/HDR packages are isolated in `../requirements-optional.txt`.
Source-preserving inference saves initial and final panoramas as PNG and lossless
8-bit RGB TIFF. TIFF output does not imply HDR support.

For AWS GPU training, configure a compatible CUDA-enabled PyTorch environment
before running the same workflow; the local CPU test does not verify that setup.

## Train AI stitching followed by correction

The shared config selects `ai_pipeline.mode: tiled_neural`. `PanoramaModel`
stitches calibrated source frames into an initial RGB panorama; the correction
pipeline processes that output next. Both stages need trained checkpoints.

Prepare a dataset with both layouts:

```text
outputs/training_bundle/
  panorama/train/<scene>/images/, capture.json, panorama.png
  panorama/val/<scene>/images/, capture.json, panorama.png
  restoration/train/<scene>/before.png, after.png, pair.json
  restoration/val/<scene>/before.png, after.png, pair.json
```

Each stitching scene needs source photos, accurate camera intrinsics/rotations
in the [shared camera contract](#shared-camera-input), and a reference
`panorama.png` aligned to that camera coordinate system. Supported pose adapters
can replace `capture.json`. For the default `rgb_residual` model, this must be a
genuine 12000x6000 reference matching `model.train_output_height/width`; smaller
targets are rejected, and enlarging a small reference does not create valid detail
supervision. Inputs stay at their original dimensions. Drone/mobile profiles
accept a long edge up to 3500 and short edge up to 2100 in either orientation,
alongside their original profile sizes. Calibration must match every source image.
Split by physical scene, keeping every variant of a
scene in one split across both datasets. The existing pair packager supplies
only `restoration/`; calibrated stitching scenes must be prepared separately.
`Stitched.jpg` / `Edited.jpg` pairs alone cannot train AI stitching.

```powershell
& $trainPython -m panorama.pano_ai.train.run_training --config panorama/pano_ai/train/configs/stitching_correction_job.yaml
```

This job trains `panorama` first and `combined` second, independently. It writes
`panorama/panorama_native_detail_best.pt` and `combined/combined_best.pt` under
`outputs/stitching_correction_training`. It does not automatically generate
correction pairs from stitching predictions. For correction training matched
to the deployed stitcher, prepare reviewed before/after pairs from its outputs
and train correction on those pairs, preserving the held-out scene split.

After validation, set these fields in `panorama/stitching/config.yaml` (paths
resolve relative to that config):

```yaml
ai_pipeline:
  mode: tiled_neural
inference:
  checkpoint: ../../outputs/stitching_correction_training/panorama/panorama_native_detail_best.pt
correction:
  checkpoints:
    combined: ../../outputs/stitching_correction_training/combined/combined_best.pt
  toggles:
    combined: true
```

Update the existing mappings, retaining their other settings and keeping the
other post-blend correction toggles false for combined correction.
Run the complete pipeline on a calibrated capture scene:

```powershell
& $trainPython -m panorama.pano_ai.tiled_inference --scene "D:/Captures/scene_001" --config ../stitching/config.yaml --output-dir outputs/ai_panorama
```

Neural stitching remains unvalidated for production quality. A checkpoint with
`contract: panorama_native_rgb_residual_v7` is required for the default detail
architecture. A new checkpoint must be trained; the repository does not supply
validated native-detail weights. Correction remains disabled
in the shared config until its trained checkpoint has been evaluated.

## Training logs and quality metrics

Each training stage prints its effective configuration, device, optimizer settings,
learning rates, dataset counts and total/trainable parameter counts before the
first epoch. Epoch summaries are flushed to the console and appended to
`<output_dir>/<task>/training_log.jsonl` (or the output directory of a standalone
trainer). Each invocation has a run ID; resumed runs append without erasing history.
The final training report links each stage's log, and checkpoints record the
selected epoch's metrics and learning rates.

Panorama and correction stages log training and validation loss, plain RGB L1,
PSNR in dB, a **global SSIM proxy**, and `pixel_accuracy_pct`. Pixel accuracy means
the percentage of pixels with **all three RGB errors <= 8/255**; it is not
classification accuracy or a guarantee of good stitching. Training metrics use
the predictions made during optimization, while validation uses the held-out
split (EMA weights for panorama when enabled). The correction training loss is
the weighted structural objective; its validation loss is plain L1, so compare
the separately logged train/validation L1 for the same error definition.
Mobile validation metrics are included when enabled. Pairwise alignment/blending
tasks log their own supervised loss and learning rate; RGB accuracy does not apply.
Learning rates are read from every optimizer parameter group each epoch; these
trainers currently use constant learning rates unless their optimizer is changed.
For native-detail stitching, training metrics cover the sampled native-scale
crops. Validation averages the top-left, top-right, bottom-left, bottom-right and
centre crops (duplicate origins are removed for small images), with EMA weights
when enabled. These are crop metrics, not full-12K quality scores. The best
panorama checkpoint uses the weighted validation objective; combined correction
selects by its own held-out L1 policy.

## What the updated stitching and correction models address

Neural stitching uses exposure-matched source tiles, fixed encoder BatchNorm
statistics, smooth tile weights, per-camera normalization and feathered camera
footprints. This targets exposure steps and tile/camera boundary artifacts. The
legacy `model.detail.mode: features` decoder reconstructs RGB from a lower-resolution
feature canvas; requesting 12000x6000 does not guarantee original photo detail.
The default `rgb_residual` architecture adds a native RGB skip and learned
detail refinement inside `PanoramaModel`, retaining `ai_pipeline.mode: tiled_neural`.
It does not call the classical stitcher. Original calibrated photos are sampled
directly onto each output tile, using the same exposure gains in training and
inference. Both branches apply the estimated gains in linear light. One source
wins each pixel by footprint quality for fine texture. Full-source Gaussian
low-frequency brightness and colour are feathered by camera footprint weights,
with a smooth chromatic-conflict guard retaining ownership for incompatible views.
This reduces tonal seams without averaging high-frequency displaced edges.
It does not solve geometric misregistration, moving objects, or all spatial
illumination differences. Filtering uses source coordinates so output tiles
and longitude halos use identical photometry. The learned head receives RGB, coverage, overlap
disagreement, and decoded scene features. Its final layer starts at zero,
preserving observed RGB exactly before learning bounded residuals
(`model.detail.residual_scale`, default 0.1). Unobserved pixels stay black.
Output halos use periodic longitude and clamped poles.

This architecture requires a new trained checkpoint with contract
`panorama_native_rgb_residual_v7`. Old feature-only checkpoints are rejected even
with `allow_legacy_checkpoint`; there is no automatic conversion to the new head.
Native v6 checkpoints also require retraining: the native RGB input now includes
deterministic photometric feathering. The legacy override cannot bypass this change.
New checkpoint names are `panorama_native_detail_best.pt` and
`panorama_native_detail_last.pt`, keeping legacy checkpoints separate. For legacy
comparisons, explicitly select `model.detail.mode: features` and the old checkpoint.

Native-detail training requires genuine reference panoramas matching the configured
output dimensions (default 12000x6000); low-resolution targets are rejected rather
than upscaled. The trainer renders `native_crop_size` (default 1024) crops at that
global pixel scale, with four crops per scene per epoch. Crops sample observed
coverage; half preferentially sample overlapping view disagreement. L1, SSIM,
perceptual, and the new `edge_weight` (default 0.2) supervise detail. Validation
reports five fixed native-scale crops per scene, not full-panorama metrics.
Encoder activation checkpointing reduces training activation memory, at a compute
cost. Full output buffers, coarse feature canvases, and source tile batches still
consume memory; validate 12K inference on the intended GPU.

Neither architecture automatically corrects translation parallax. Hard ownership
can expose tonal or geometric seams; residual cleanup must be trained on reviewed,
aligned targets. Preserve native pixels and calibrated intrinsics rather than
pre-upscaling or heavily sharpening inputs. Compare trained results using identical
held-out 12K crops before claiming improved sharpness or seam quality.

## Native-detail configuration and checkpoint migration

The detailed [neural architecture](../ARCHITECTURE_VARIABLE_TILED.md#1-neural-stitching-with-native-rgb-detail)
documents tensor shapes, ownership, residual prediction and output halos.
The following settings already exist in `panorama/stitching/config.yaml`:

```yaml
model:
  output_height: 6000
  output_width: 12000
  train_output_height: 6000
  train_output_width: 12000
  detail: {mode: rgb_residual, residual_scale: 0.1}
training:
  native_crop_size: 1024
  native_crops_per_scene: 4
  seam_crop_probability: 0.5
  checkpoint_encoder: true
loss:
  supervised: {l1_weight: 1.0, ssim_weight: 0.2, perceptual_weight: 0.1, edge_weight: 0.2}
ai_pipeline:
  mode: tiled_neural
```

Merge changes into the existing mappings; this excerpt is not a complete config.
Train only the panorama model with:

```powershell
& $trainPython -m panorama.pano_ai.train_tiled --config ../stitching/config.yaml
```

The standalone trainer and `tiled_inference` CLI resolve their config relative
to `panorama/pano_ai/`. Default standalone data folders are `data/train` and
`data/val` under that directory, each containing scene folders. To use the two-stage bundle
layout, use `train.run_training` as above or set absolute training/validation
paths. The job runner resolves its job paths relative to the job YAML; the
stitching interface resolves `--config` from the current working directory.
Explicit `--checkpoint` paths are resolved from the working directory; configured
`inference.checkpoint` and correction checkpoint paths resolve from the shared
config directory. Neural dimensions come from `model.output_height/width`;
`stitching.output_height/width` controls classical dispatch, not neural rendering.

For an intentional old-checkpoint comparison, set `model.detail.mode: features`,
restore the old training dimensions if retraining that architecture, and point
`inference.checkpoint` to the v4 checkpoint. `allow_legacy_checkpoint` does not
bypass a detail-mode or native residual-scale mismatch. Keep exposure settings
consistent with checkpoint configuration. Training the separate combined model
does not create the v7 panorama checkpoint.

Before full training, validate scene alignment, source calibration and native
reference dimensions; split by physical scene and run the native-detail tests
below. Do not pre-upscale sources or assume the synthetic mobile correction
generator creates genuine 12K stitching supervision. Reduce tile batch size or
training crop size when GPU memory is limited, while retaining the 12K coordinate
lattice and genuine 12K reference. Decoder output tiles must exceed the fixed
64-pixel output overlap. Activation checkpointing does not bound all scene memory.

New combined correction checkpoints use `paired_panorama_restoration_v2`:

- A wider-context residual U-Net starts as an exact identity. Its dilated
  bottleneck can learn spatial seam cleanup, local deghosting, denoising and
  detail restoration, as well as color changes, from reviewed target examples.
- Spatial normalization is removed to avoid tile-dependent brightness changes.
- The objective combines pixel fidelity, local SSIM, edge gradients and coarse
  scales. Edited regions receive extra weight while unchanged pixels remain
  supervised. Configure weights under `combined.loss` in the training job.
- Half the training crops are sampled around local before/after edits when
  available (`combined.changed_crop_probability`); the rest remain uniform.
  Validation always uses the same fixed crops, without target-driven sampling.
- Inference adds 160 pixels of context on each side, wraps longitude, clamps
  poles, aligns the pooling lattice and blends floats before one quantization.
  Context padding increases inference memory compared with v1.
- Reports include an unchanged-input paired-validation baseline and
  `paired_beats_baseline_l1`; inspect visual edge/texture/join metrics too. Best
  checkpoints are selected by held-out L1, not the weighted training objective.

Existing v1 correction checkpoints still load through the original model for
inference. V2 training starts fresh; v1 resume/fine-tuning is rejected because the
architecture differs. V2 resume also requires the same loss settings. Fine-tune
from a v2 checkpoint when intentionally changing the objective.

Correction does not have source frames, depth or alternative views. It cannot
reliably reconstruct missing content, clipped highlights or fix large geometric
misregistration. Include actual stitching failures and reviewed repairs in the
pairs: seam/exposure changes, ghost removal, noise/detail cleanup, glare and
small missing regions where a trustworthy target exists. Pairs containing only
color edits cannot teach those other repairs. Keep the panorama coordinate
system aligned and physical scenes separated across train/validation.

The two-stage job does not regenerate correction inputs automatically. After
training the new stitcher, render its outputs, prepare/review the matching
before/after pairs, then run a `tasks: [combined]` job on those pairs. Validate
initial and corrected panoramas on unseen captures before enabling correction.

## 1. List three real scene pairs for the first trial

### Scan your existing Drone / DSLR sample folders

For `Files/<year>/<location>/<scene>/` containing `images/`, `Stitched.jpg`,
`Edited.jpg` and optionally `Panorama.pts`, generate the supporting files:

```powershell
& $trainPython -m panorama.pano_ai.data.scan_samples --input "D:/Files" --out outputs/sample_training_setup
```

The input can also be `D:/Files/2026` or `D:/Files/2026/Location`.
The scanner recursively finds scene folders and writes `pairs.json`,
`training_job.yaml` and `scan_report.json` into a new output folder. It does
not modify your samples. Incomplete scene folders fail explicitly.

Review `pairs.json`: set each `domain` to `dslr` or `drone` (initially null),
then verify alignment as described in step 2 before setting
`alignment_verified` to true. If every scene has the same source, you can pass
`--domain dslr` or `--domain drone` to prefill it.
The scanner assumes the panoramas are equirectangular; confirm they are full
2:1 spheres. Dimensions and orientation are checked during packaging.

Splits are assigned by location name, with the last roughly 20% of sorted
locations reserved for validation (at least one when multiple locations exist).
The same location name across years stays together. Review these identities:
merge aliases for the same physical location, and distinguish unrelated locations
with identical names if needed. A single location produces only training entries;
add pairs from an independent validation location before starting training.

After review, package and train using the generated YAML:

```powershell
& $trainPython -m panorama.pano_ai.data.prepare_pairs --manifest outputs/sample_training_setup/pairs.json --out outputs/sample_training_setup/training_bundle
& $trainPython -m panorama.pano_ai.train.run_training --config outputs/sample_training_setup/training_job.yaml
```

The generated YAML points to that bundle and saves results under
`outputs/sample_training_setup/training_run`. It retains the template's training
settings; adjust epochs/crop size for your intended run as described below.
Use a new setup output folder for subsequent scans.

For this **combined panorama correction** training, only `Stitched.jpg` and
`Edited.jpg` are consumed. No `capture.json` is needed, and `images/` and
`Panorama.pts` are not used as training inputs. Training stitching geometry
from source photos requires a separate calibrated supervision workflow.

### Alternatively, enter pairs manually

Copy the editable example:

```powershell
Copy-Item panorama/pano_ai/train/configs/pairs.example.json panorama/pano_ai/train/configs/my_pairs.json
```

Open `my_pairs.json` and replace the six example image paths with your actual
files. Use forward slashes in JSON paths, e.g. `D:/PanoramaData/room/ptgui.png`.
Each entry has:

| Field | What to enter |
| --- | --- |
| `scene_id` | Unique name for this pair, using letters/numbers/underscore/hyphen |
| `source_scene_id` | Physical scene/location identity; all variants keep this ID |
| `before` | PTGui panorama before your corrections |
| `after` | Final corrected panorama of the same scene |
| `split` | `train` for two locations, `val` for a different location |
| `domain` | `dslr`, `drone`, or `real_mobile`, matching the actual source |
| `projection` | `equirectangular` for a full spherical panorama |
| `alignment_verified` | Change `false` to `true` only after the review below |

You can add more entries later. Split by physical scene/location, not by crop or
filename. Keep exposure brackets, alternate edits and generated variants in
the same split. The example paths are placeholders and cannot train as supplied.

## 2. Review alignment once for each pair

Check that before/after show the same scene at the same heading, crop and pixel
coordinates. Both must be complete 2:1 spherical panoramas with identical
dimensions and normalized orientation. Use consistent color space; lossless
PNG/TIFF is preferable where available. This pipeline converts inputs to 8-bit RGB.

Do not use partial drone panoramas, orthomosaics, or HDR files without preparing
them appropriately first. If a correction shifted/resized the whole panorama,
align it before marking the pair verified. The preparation command does not
perform image registration or prove alignment automatically.

The default training crop is 1024x1024, so images must be at least 1024 pixels
high. You do not need defect masks or camera poses for this combined correction
training route.

## 3. Automatically package the dataset

```powershell
& $trainPython -m panorama.pano_ai.data.prepare_pairs --manifest panorama/pano_ai/train/configs/my_pairs.json --out outputs/training_bundle
```

This validates all entries before copying, preserves your originals, writes
portable relative image paths and creates:

```text
outputs/training_bundle/
  preparation_report.json
  restoration/
    train/<scene>/before.png, after.png, pair.json
    val/<scene>/before.png, after.png, pair.json
```

Original image extensions are preserved (the files may be `.jpg` or `.tif`).
Use a new output folder for subsequent preparation runs; the command refuses
to overwrite a nonempty dataset. It copies images, so allow disk space for them.

## 4. Point the training config to the prepared data

Edit `panorama/pano_ai/train/configs/training_job.yaml`. Update these existing
settings, retaining the remaining fields:

```yaml
dataset_root: ../../../../outputs/training_bundle
output_dir: ../../../../outputs/training_pilot
epochs: 2
tasks: [combined]
mobile_validation:
  enabled: false
  path: restoration/mobile_val
```

Paths in this YAML resolve from its own directory, not the terminal directory.
The relative dataset path above points to the output of step 3. An absolute path
such as `D:/PreparedDatasets/panorama-v1` also works.

For a quick CPU execution trial, temporarily use `combined.input_size: [256, 256]`
and `combined.crops_per_scene: 4`. Keep `batch_size: 1`. This smaller run tests
training; it is not the final production training setup.

## 5. Start training

```powershell
& $trainPython -m panorama.pano_ai.train.run_training --config panorama/pano_ai/train/configs/training_job.yaml
```

The terminal prints training and validation loss each epoch. Results go to:

```text
outputs/training_pilot/
  training_report.json
  combined/
    training.yaml          # Effective settings for this run
    combined_best.pt       # Best checkpoint by the selected validation loss
    combined_last.pt       # Latest completed epoch, including earlier best checkpoint
```

Check that the report has a finite validation loss and a positive
`parameter_change_l1`. Then compare actual corrected images on held-out scenes;
falling loss alone does not establish better stitching.

For a larger run, use a new `output_dir`, restore `input_size: [1024, 1024]` and
`crops_per_scene: 12`, and set the intended epoch count (e.g. the original 20).
By default each invocation starts afresh. To resume combined/restoration training,
set `combined.resume` to a trusted local `combined_last.pt` (or `combined_best.pt`) checkpoint and set
`epochs` to the desired **total** epoch count (greater than the checkpoint epoch).
Relative resume paths resolve from the job YAML directory. Use a new output folder
and keep the dataset, splits, crop settings, batch size and checkpoint-selection
policy unchanged.

```yaml
combined:
  resume: ../../../../outputs/training_pilot/combined/combined_last.pt
  # Retain the other combined settings.
```

Resume restores weights, optimizer (including its learning rate), completed epoch
and available random-number state. It continues from the epoch in the chosen file. `combined_last.pt` records every
completed epoch and embeds the best checkpoint, so recovery into a new output
folder preserves the earlier best even if later validation scores worsen.
Checkpoint files are replaced atomically after writing; an interrupted epoch must
be repeated. Latest checkpoints use additional disk space because they embed the best. Older checkpoints without random state
can resume but cannot reproduce the original sampling sequence. Only load trusted
checkpoint files. Cross-device bitwise equivalence is not promised. Panorama and
pairwise trainers reject resume/fine-tuning settings. Resume keeps the previous
optimizer; use fine-tuning below when starting a new adaptation run.

Do not treat 20 epochs as a convergence guarantee. Keep production correction
disabled until validated.

### Fine-tune a restoration model

Set `combined.finetune` instead of `combined.resume`, use a new `output_dir`,
and retain the checkpoint's `base_channels` and task:

```yaml
combined:
  finetune: ../../../../outputs/training_pilot/combined/combined_best.pt
  lr: 0.00002
  # Keep input_size, base_channels, batch_size and other desired settings.
```

This loads only model weights and starts epoch numbering, optimizer, random
sampling and best-score selection afresh. `epochs` specifies the number of new
training epochs. You may supply new reviewed training pairs, but keep validation
and test scenes independent of both the original training data and new data.
Existing split checks cover the current bundle; they cannot detect historical
pretraining leakage. Resume and fine-tune cannot be enabled together. Checkpoints
must be trusted local files with matching task, contract and channel count;
relative paths resolve from the job YAML directory. The checkpoint and training
report record `finetuned_from` for provenance.

## 6. Add real-mobile validation when available

Add reviewed mobile pairs to your manifest with `split: mobile_val`,
`domain: real_mobile` and new physical-scene IDs absent from train/val. Their
before images should come from the mobile stitching pipeline, with aligned
reviewed corrected targets. Package the expanded manifest to a new dataset
folder, update `dataset_root`, then set:

```yaml
mobile_validation:
  enabled: true
  path: restoration/mobile_val
```

Checkpoint selection then defaults to real-mobile validation loss. The report
also compares against the uncorrected baseline and records edge, texture and
longitude-join metrics on fixed native crops. Full-panorama visual review is
still required. Leave this disabled while real reviewed mobile pairs are absent;
synthetic examples must not be labelled real mobile.

## 7. Optional: automatically generate mobile-style examples

This route needs reviewed final panoramas, not before/after pairs. Create
`references.json` with train/val locations:

```json
{
  "version": 1,
  "scenes": [
    {"scene_id": "room_a", "source_scene_id": "location_a", "panorama": "D:/FinalPanoramas/a.png", "split": "train", "reviewed": true, "projection": "equirectangular"},
    {"scene_id": "room_b", "source_scene_id": "location_b", "panorama": "D:/FinalPanoramas/b.png", "split": "val", "reviewed": true, "projection": "equirectangular"}
  ]
}
```

```powershell
& $trainPython -m panorama.pano_ai.data.prepare_mobile --manifest references.json --out outputs/mobile_training_bundle --variants 3 --output-width 2048
```

The generator automatically renders calibrated perspective frames, adds
controlled exposure/blur/noise/JPEG changes, stitches them, and writes training
pairs. Point `dataset_root` to `../../../../outputs/mobile_training_bundle` and
run step 5. Do not run `prepare_pairs` on this output; it is already packaged.

To use camera intrinsics/rotations from your existing 1920x1080 mobile capture,
add `--template "<mobile scene folder>" --frame-width 640 --frame-height 360`.
Only calibration is reused; no mobile image content becomes a target.

For mixed training, copy the generated scene directories from
`restoration/train` and `restoration/val` into the corresponding prepared-pair
folders. Use distinct pair names, preserve `source_scene_id`, and keep each
physical location in only one split across both sources. The loader rejects
cross-split physical-scene overlap. Do not move synthetic scenes into mobile_val.

Generated views simulate rotation and image degradation, not real translation
parallax or hidden surfaces. See [mobile generation and validation details](#mobile-generation-and-validation-details) for
the exact generator/validation behavior.

## What is automated?

| Work | Automated? |
| --- | --- |
| Match the correct before/after files and confirm visual alignment | No: supply/review the manifest |
| Check dimensions, orientation, domains and split leakage | Yes |
| Copy originals into the training layout and write pair records | Yes: `prepare_pairs` |
| Render mobile-style captures and synthetic correction pairs | Yes: `prepare_mobile` |
| Native crop sampling, training, validation and checkpoint saving | Yes: `run_training` |
| Prove real mobile quality or correct badly misaligned pairs | No |

## While the real dataset is unavailable

Repeat the synthetic execution test with a new output folder:

```powershell
& $trainPython -m panorama.pano_ai.train.smoke_paired --out outputs/paired_smoke_next
```

This checks execution without implying real-data convergence.

## AWS and S3

Upload the complete prepared folder tree, then change `dataset_root` in
[training_job.yaml](train/configs/training_job.yaml):

```yaml
dataset_root: s3://your-bucket/datasets/panorama-v1/
```

The S3 adapter lists and downloads the dataset to `cache_dir`, preserving its
layout. Cache snapshots are keyed by object keys, sizes and ETags; changing a
dataset creates a separate snapshot. Use an immutable/versioned prefix during
training. This is local staging, not on-demand S3 streaming, so sufficient local
disk for the dataset, cache and checkpoints is required.

After the runtime, dataset layout and IAM role are configured, switching the
dataset input requires only changing `dataset_root`. AWS setup still needs:

- The repository and compatible PyTorch/CUDA dependencies installed.
- S3 read/list permissions and credentials through the standard AWS role or
  credential chain; no credentials are stored in this YAML.
- Appropriate writable `cache_dir` and `output_dir`, available disk and GPU memory.

Outputs are written to `output_dir`. S3 checkpoint upload, managed job creation
and automatic resume are not included. The S3 adapter is unit-tested with a fake
client; a live S3 account was not used for the local smoke test. AWS documents
[Boto3 downloads](https://docs.aws.amazon.com/boto3/latest/guide/s3-example-download-file.html)
and the [credential chain](https://docs.aws.amazon.com/boto3/latest/guide/credentials.html).

## Technical reference (optional)

The following details are not prerequisites for the combined training workflow.

<details>
<summary>Mobile adaptation, inference, camera formats, optional models and evaluation</summary>

## Mobile generation and validation details

Step 7 requires reviewed, orientation-normalized, full-sphere 2:1 reference
panoramas. Relative image paths resolve from the manifest. `source_scene_id`
identifies the physical location across cameras, edits and generated variants;
omit it only when `scene_id` already provides that identity.

The generator renders 26 pinhole views per variant by default, records their
calibration, applies seeded exposure/noise/blur/JPEG changes, and runs the actual
geometric stitcher to produce `before.png`. The resized clean reference becomes
`after.png` in the same spherical coordinates. Variants retain the physical
scene ID and are labelled `domain: synthetic_mobile`. Outputs are:

```text
captures/{train,val}/<scene_variant>/capture.json, images/...
restoration/{train,val}/<scene_variant>/pair.json, before.png, after.png
generation_report.json
```

With `--template`, the frame aspect ratio must match the capture template.
Intrinsics scale using pixel-centre conventions; rotations are retained but
translations are zero because the rendered views share one optical centre.
A 2048-wide reference supports the default 1024x1024 native training crop;
smaller smoke-test outputs require smaller crops.

These examples do not simulate translation parallax, rolling shutter or real
moving people. Their bounded corruptions are not a measured phone-camera noise
model. Known reference pixels can supervise synthetic holes but do not establish
recovery of hidden real surfaces. The compositor works at at most 4096 pixels
wide. Inputs are converted to 8-bit RGB; normalize HDR and color space first.

Real mobile validation pairs live under
`<dataset_root>/restoration/mobile_val/<scene>/pair.json`, with the same pair
fields used in step 1 and `domain: real_mobile`. Their physical locations must
be absent from both training and ordinary validation. Synthetic and unknown
domains are rejected. Domain and alignment fields declare reviewed provenance;
they do not automatically classify images or verify visual alignment.

When mobile validation is enabled, checkpoint selection defaults to mobile L1.
Set `combined.checkpoint_selection: paired` to explicitly select ordinary
validation instead; `mobile` requires mobile validation data and fails without
it. Checkpoints and `training_report.json` record the selection score, separate
mobile metrics, uncorrected baseline metrics and `mobile_beats_baseline_l1`.
Lower L1 does not imply improvement in every visual metric.

Evaluation uses fixed native crops. Longitude error is measured only when a
crop crosses the actual panorama join, not between arbitrary crop edges.
Inspect full panoramas and keep an additional real-mobile test set separate
from the validation scenes used to select checkpoints.

Verify the implementation with:

```powershell
& $trainPython -B -m unittest panorama.pano_ai.tests.test_mobile_adaptation
```

The existing local run in `outputs/mobile_adaptation_smoke/` generated 78
calibrated frames and three correction pairs from three synthetic references,
then trained the combined model for two epochs. Its report has no real-mobile
metrics. Unit fixtures test mobile checkpoint selection but are not a real
mobile benchmark. Real validation stays disabled until reviewed pairs exist.

## Inference modes

The alternative `ai_pipeline.mode: source_preserving` in
[`../stitching/config.yaml`](../stitching/config.yaml) projects calibrated source
RGB, optionally refines views with separately trained alignment/ownership heads,
then runs explicitly enabled restoration stages. With learned heads disabled it
uses geometric projection and graph-cut/multiband composition. It does not load
the legacy panorama decoder checkpoint in this mode.

The default `ai_pipeline.mode: tiled_neural` runs the learned stitching model
with `model.detail.mode: rgb_residual` first and correction second. Its checkpoint
contract is `panorama_native_rgb_residual_v7`. The explicit feature-only detail
mode retains `panorama_exposure_blending_v4`. Architecture/residual-scale
mismatches fail before loading weights; the legacy flag cannot convert weights.
Feature fusion uses smooth tile weights, normalizes all tiles of each camera,
then feathers camera footprints. Tile chunks from each camera must be contiguous.
Feature sampling clamps to the border to avoid darkening the final stride interval.
Exposure matching estimates bounded scalar gains in linear light from calibrated
overlap previews, rejecting clipped/dark pixels and inconsistent overlaps. It runs
identically during training and inference; no ground-truth panorama is used.
`input.exposure_compensation` controls this stage and must match the checkpoint.
Metadata records gains, accepted overlap pairs and before/after log mismatch.
Isolated frames keep unity gain. Scalar gains do not correct spatial vignetting or
white-balance differences. Validate on held-out captures before deployment.
`inference.use_ema` selects EMA weights when present, otherwise model weights;
metadata records which weights were loaded. Correction checkpoints also honor
their section's `use_ema` setting and stored channel count.

## Shared camera input

Both calibrated classical stitching and AI consume `stitching.capture`.
Existing `images/ar_poses.jsonl` and legacy `poses.pt` + `camera.json` are adapted.
A canonical scene may instead provide `capture.json`:

```json
{
  "version": 1,
  "coordinates": "opengl_camera_to_world",
  "source": "calibrated",
  "frames": [{
    "name": "camera_0",
    "filename": "images/frame_0.jpg",
    "width": 2048,
    "height": 2048,
    "intrinsics": [1200, 1200, 1024, 1024],
    "rotation": [[1,0,0],[0,1,0],[0,0,1]],
    "translation": [0,0,0],
    "projection": "pinhole",
    "depth": {"path": "depth/frame_0.npy", "format": "npy_meters", "units": "metres"}
  }]
}
```

Intrinsics are `[fx, fy, cx, cy]` in original sensor pixels. Do not EXIF-rotate
images without transforming calibration. Rotation maps camera to world; camera
axes are right/up/backwards (+X,+Y,+Z). Translation is in metres. `depth` is
optional; `fisheye_180` additionally requires `fov_degrees`. Filename paths are
relative to the manifest. Classical filenames must have unique stems. Frame
counts and resolutions still have to match an input profile in config.

Translations and depth metadata are preserved, **but composition currently uses
rotation-only geometry**. Raw ARCore `.depth` files remain opaque. This change
does not implement depth fusion or reconstruct surfaces hidden by the person.
Original ARCore manifests are retained for guided heading recovery when present.

## Uncertainty and fallback

Enable trained heads with `ai_pipeline.learned_views.alignment.enabled` and
`blending.enabled` in the alternative `source_preserving` route. These settings
do not activate alignment inside the neural `rgb_residual` decoder.
Each requires its own checkpoint with the matching task and
`geometric_pairs_v1` contract. Missing or mismatched weights fail explicitly.
Alignment uses a 1024x512 spherical canvas and inverse displacement in those
canvas pixels. Confidence, displacement bounds, forward/backward consistency,
Jacobian bounds, textured overlap and photometric improvement all gate proposals.
Rejection returns zero displacement and a reason in geometry metadata. Original
coverage is retained if a local warp would expose a hole. Blending selects only
confident valid source owners; other pixels retain geometric seam ownership.
These checks are safeguards, not a calibrated probability or a quality guarantee.

Keep old `advanced_corrections` pairwise toggles disabled: use the source-view
heads for geometric alignment. Post-blend RGB correction cannot recover camera
geometry. The learned heads are disabled by default because no validated trained
weights are supplied.

## Defect masks

For deterministic photographer-shadow lightening on a finished panorama, use
an exact-size grayscale `shadow_mask.png`: white marks only the shadow and
black protects everything else. Do not select pixels merely because they are
dark. Feather the mask edges where practical.

```powershell
& $trainPython -m panorama.pano_classical.shadow_removal --image "D:/Scene/images/classical_panorama.jpg" --mask "D:/Scene/shadow_mask.png" --out outputs/shadow_cleanup/corrected.png
```

This estimates surrounding illumination and applies a bounded RGB gain within
the mask, retaining texture and preserving unmasked pixels exactly. It is
approximate on mixed surfaces and cannot remove body fragments or recover
detail from black/clipped pixels. Inspect the result before using it as a target.
For the classical stitching route, set `classical_finishing.shadow_removal: true`
and put `shadow_mask.png` in the scene folder (the parent of `images/`), or add
`"shadow": "shadow_mask.png"` to the existing defect manifest. The saved
`01_shadow_removal.png` preserves unmasked pixels; subsequent global color and
sharpen stages may change them. The AI route does not use this new stage.

Create `defects.json` in the scene with masks at the exact final panorama size:

```json
{
  "version": 1,
  "coordinates": "equirectangular",
  "size": [12000, 6000],
  "masks": {
    "photographer": "photographer_mask.png",
    "moving_objects": "ghost_mask.png",
    "glare": "glare_mask.png"
  }
}
```

Categories are `missing_coverage`, `moving_objects`, `photographer`, `glare`,
`lens_dots`, `other`. Black is protected, white is repair, intermediate values
blend repairs. Masks must match output heading, crop and dimensions. Actual
geometric coverage supplies `missing_coverage`; black RGB, bright lights and
valid floor/ceiling are not automatically defects. Mask files and fractions are
saved under `defect_masks/` with the result. Without a manifest, explicitly named
`correction_mask.png`, `ghost_mask.png`, `photographer_mask.png` are supported.

AI `nadir_zenith` restoration consumes missing/photographer/other masks;
`ghost_removal` consumes moving-object masks; glare/dots use their own masks.
Masked stages preserve unmasked pixels exactly. Global color/sharpen stages
intentionally change the image, so evaluate preservation at the masked stage's
saved output, before those stages. Classical small-defect inpainting consumes
the same masks, subject to its 1% area limit. Existing scene-specific classical
object-removal polygons remain separate. There is **no automatic person detector**.

## Optional separate-stage training

All commands run from the repository root with torch/torchvision installed.

```powershell
python -m panorama.pano_ai.train.train_pairwise --task alignment --train data/pairs/train --val data/pairs/val --out panorama/pano_ai/checkpoints
python -m panorama.pano_ai.train.train_pairwise --task blending --train data/pairs/train --val data/pairs/val --out panorama/pano_ai/checkpoints
python -m panorama.pano_ai.train.train_restoration --stage nadir_zenith --data data/restoration --config restoration_training.yaml --out panorama/pano_ai/checkpoints
python -m panorama.pano_ai.train_tiled --config ../stitching/config.yaml
```

Pair samples are `.npz` files with scalar string `contract=geometric_pairs_v1`;
`reference`, `source` are RGB float [0,1] arrays `[512,1024,3]` produced by the
same calibrated projection/exposure stage. `reference_valid`, `source_valid`,
`confidence`, `supervision_valid` are `[512,1024,1]` float masks [0,1]. Alignment
additionally needs ground-truth `flow` `[512,1024,2]`, inverse `[dx,dy]` in pixels
(norm <=8). Blending needs ground-truth `weight` (1 selects reference, 0 selects
source). Confidence labels must mark trustworthy correspondence/ownership;
do not label every overlap reliable. Blending labels should be prepared after
alignment using the geometric seam reference used at inference. Split by scene,
never by overlapping crops of the same scene. The trainer does not fabricate
these ground-truth labels from an unverified panorama.

Restoration scenes contain `stages/<stage>/{before.png,after.png,mask.png}`.
For these separate-stage datasets, all stages except global `color` require an explicit mask. The recommended `combined` task uses the reviewed pairs described above instead. Training uses native
paired crops, including wrapped longitude crops; it never stretches the whole
panorama down to a crop. A minimal training YAML is:

```yaml
input_size: [1024, 1024]  # Match inference tile size; images must be at least this size.
base_channels: 32        # Multiple of 8 for GroupNorm.
batch_size: 1
epochs: 20
lr: 0.0001
```

For masked stages, restoration loss is restricted to masked pixels. Validation also reports edge,
texture, longitude-join and unmasked-preservation metrics. Its deterministic
crop samples the middle positive mask point; prepare multiple paired samples to
cover large/multiple defects. Use a held-out full-scene visual regression run
before enabling a trained stage. Panorama RGB training honors L1, SSIM,
perceptual and edge weights; nonzero unsupported geometry losses fail instead of being
silently ignored. Perceptual loss uses pretrained ResNet18 weights (download
required if uncached); setting its weight to zero avoids loading that network.

## Regression evaluation

```powershell
python -B -m unittest panorama.pano_ai.tests.test_regressions panorama.pano_ai.tests.test_contracts
python -B -m unittest panorama.pano_ai.tests.test_native_detail
python -B -m unittest discover -s panorama/pano_ai/tests
python -B -m unittest discover -s panorama/pano_classical -t . -p "test_*.py"
python -m panorama.stitching.visual_regression --candidate corrected.png --reference ground_truth.png --original initial.png --mask photographer_mask.png --out outputs/visual_check
```

Evaluation requires identical coordinates and resolution. It saves JSON metrics
and native centre/bottom/longitude-join comparison crops (reference on left).
`--limits limits.json` exits nonzero for exceeded metric maxima, e.g.
`{"unmasked_changed_pixels": 0, "longitude_join_error": 0.02}`. Set thresholds
from held-out captures; there is no universal good-image threshold. Comparing
to a baseline measures differences, not proof of improvement. Use lossless PNG
for exact unmasked preservation; JPEG recompression changes protected pixels.

Native-detail tests verify source texture preservation at initialization,
single-source ownership of conflicting views, overlap disagreement, periodic
projection, learned tile/crop continuity, real head/encoder gradient updates,
target-resolution rejection, checkpoint training/reload and mismatch rejection.
A dedicated crop test uses the actual 12000x6000 coordinate lattice. These are
synthetic execution checks, not a full-size GPU benchmark or real-scene quality
study. Compare trained 12K outputs against aligned references and the explicit
v4 feature-only baseline at identical dimensions.

Only the alternative `source_preserving` and calibrated classical composition
routes cap working width at 4096 and upscale larger outputs. The default native
neural detail path samples source RGB directly at requested output coordinates.
Larger output dimensions alone cannot restore fine detail or motion blur.


## Complete separate-model execution test

The broader synthetic smoke test exercises seven legacy/separate models,
checkpoint reload, EMA selection and masked preservation:

```powershell
python -B -m panorama.pano_ai.train.smoke_training --out outputs/ai_training_smoke_new
```

Use a new output folder. The test writes `smoke_report.json`, checkpoints and
inference images there. Optional `--validate-captures <scene1> <scene2>` checks
real capture metadata without training on invented labels. CPU tests do not
verify CUDA, live AWS permissions, or production-resolution GPU memory. The
runner uses one device and does not implement distributed training.

## Quality target: comparable to Travvir

[Travvir's public documentation](https://travvir.com/docs) describes a mobile
capture SDK and server-side processing. It does not provide a reproducible
training recipe or a benchmark that establishes parity for this implementation.
Its [published tours](https://travvir.com/explore) are a visual reference.

To assess parity, obtain authorized reference exports and compare the same scenes,
capture conditions, output orientation and effective resolution. Use held-out
rooms and outdoor scenes with nearby furniture, low texture, moving people,
bright windows, detailed floors and ceiling lines. Compare native crops and an
interactive 360 viewer for:

- Straight-edge continuity, duplicated objects and parallax.
- Fine texture, blur, exposure transitions and halos.
- The left/right longitude join and genuine missing coverage.
- Photographer removal with valid floor/ceiling preserved.
- Exact preservation outside each repair mask.

Use `panorama.stitching.visual_regression` for repeatable metrics/crops and human
review for perceptual/structural quality. Require improvement over the classical
baseline on held-out real captures before activating learned heads. Two or three
scenes can test execution and memorization; they cannot prove generalization.
No fixed scene count or number of epochs guarantees this quality target. The
alternative geometric compositor's 4096-pixel working-width cap also limits
effective detail when its output is enlarged to 12000 pixels. The native neural
skip avoids that cap but cannot recreate detail absent from the source photos.

</details>

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
