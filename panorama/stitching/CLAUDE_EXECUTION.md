# Panorama training and inference execution guide

Follow this guide in order on the GPU instance. Execute the work, verify each
stage, and record the results. Do not treat the example paths as existing assets.
Replace them with confirmed absolute paths before running commands. Run Python
commands from the repository root in the activated training environment.

## Objective

Train the native RGB stitching model at the 12000×6000 coordinate scale, evaluate
its seams independently, then train and evaluate a separate combined correction
model using actual outputs from the selected stitcher.

Preserve full-resolution photos, calibration, references and previous runs.
Do not delete DNGs, upscale low-resolution targets, overwrite previous datasets,
or claim improvements from synthetic checks alone. Do not infer checkpoint
compatibility from experiment labels such as “v5”; inspect the stored contract.

## 0. Resolve paths and record the execution plan

Fill these values with paths on the instance:

| Item | Confirmed absolute path |
| --- | --- |
| Repository root | |
| Python executable/environment | |
| Raw scenes root | |
| Prepared stitching bundle | |
| Separate test scenes, if available | |
| Shared pipeline config | |
| Stitching-only job YAML | |
| Stitching run directory | |
| Stitching-only inference config | |
| Stitching-only inference output root | |
| Reviewed correction targets | |
| Local AI correction manifest | |
| Correction bundle | |
| Correction job YAML | |
| Correction run directory | |
| Corrected inference config | |
| Corrected inference output root | |

Use new directories for datasets, diagnostics and training/inference runs. Keep
all downloaded and original files. Record the selected seed, epochs, GPU and
data split. Ask for missing access or unavailable reviewed targets when needed;
continue checks that do not depend on them.

## 1. Verify the GPU instance and code

1. Synchronize the latest authorized repository changes to the instance,
   including seam sampling/loss/metrics, diagnostics and correction-pair checks.
2. Record the branch, Git commit and working-tree changes. Do not discard local
   work to synchronize the checkout.
3. Activate the intended Python environment. Verify PyTorch CUDA availability,
   GPU name, VRAM, Python/PyTorch versions and available disk space.
4. Verify dependencies, including `rawpy` when decoding DNGs. Ensure pretrained
   encoder/perceptual weights are cached or downloadable.
5. Run the regression suite and retain its output:

```bash
python -m unittest discover -s panorama/pano_ai/tests
```

6. Check the CLI help on this checkout if any command differs from this guide.
   Stop dependent training if regression tests fail.

## 2. Prepare the stitching dataset

1. Download source photos, `Panorama.pts` and the unedited `Stitched.jpg` for each
   scene using the existing authorized dataset manifest and storage access.
2. Keep source photos at full resolution. DNG decoding must use
   `half_size=False`. Preserve source orientation consistent with calibration.
3. Keep genuine 12000×6000 references; do not resize a 3000×1500 target upward.
4. Split by physical site. All variants and scenes of one site stay in one split.
   Keep a separate untouched test split if enough sites are available.
5. Prepare each scene into its chosen split. Use exactly one calibration source.

PTGui JSON import:

```bash
python -m panorama.pano_ai.data.prepare_native \
  --scene /data/raw/scene_0001 \
  --output /data/bundle/panorama/train/scene_0001 \
  --pts /data/raw/scene_0001/Panorama.pts
```

Or verified existing calibration:

```bash
python -m panorama.pano_ai.data.prepare_native \
  --scene /data/raw/scene_0001 \
  --output /data/bundle/panorama/train/scene_0001 \
  --capture /data/calibration/scene_0001/capture.json
```

6. Review `preparation.json`: calibration must be verified. The importer accepts
   a restricted PTGui feature set; stop on unsupported features or failed checks.
   Do not bypass checks or silently guess calibration.
7. Verify prepared image dimensions against `capture.json`. Inspect representative
   source projections, coverage and structural alignment, not only control-point
   statistics. Verify orientation and reference coordinate alignment.
8. Verify that source profiles accept actual dimensions, projection and frame
   counts, including 33-frame scenes. Set explicit appropriate profile limits.
9. Check every scene has supported RGB images, calibration and the full-size
   reference. Record missing/rejected scenes and reasons.

Required bundle layout:

```text
/data/bundle/
  panorama/
    train/<scene>/images/...
    train/<scene>/capture.json
    train/<scene>/Stitched.jpg
    val/<scene>/images/...
    val/<scene>/capture.json
    val/<scene>/Stitched.jpg
```

The loader also accepts `panorama.png`. Do not use `Edited.jpg` automatically as
stitching supervision. JPEG preparation is not lossless; preserve the original
photos and verify the current preparation script's encoding behavior.

## 3. Configure stitching training

1. Create a run-specific copy of
   `panorama/pano_ai/train/configs/stitching_correction_job.yaml`.
2. Set absolute `dataset_root`, `pipeline_config` and a new `output_dir`.
3. Set `tasks: [panorama]` so correction is not trained prematurely.
4. Confirm the effective model/loss configuration includes:

```yaml
model:
  output_height: 6000
  output_width: 12000
  train_output_height: 6000
  train_output_width: 12000
  detail:
    mode: rgb_residual
    residual_scale: 0.1
loss:
  supervised:
    l1_weight: 1.0
    ssim_weight: 0.2
    perceptual_weight: 0.1
    edge_weight: 0.2
    seam_weight: 0.2
```

These values already exist in the intended local config. Verify rather than
overwriting unrelated settings. Neural dimensions come from `model`, not just
`stitching.output_width/height`.

5. Retain coverage masking, boundary-focused crop sampling and activation
   checkpointing. Record crop size/count, batch size, optimizer and EMA settings.
6. Keep corrections disabled for stitcher evaluation and exposure settings
   consistent between training and inference.
7. Verify scheduler/early-stopping options are actually implemented in the running
   trainer before relying on YAML fields. Do not assume unused keys take effect.
8. Run a small pilot in a separate output directory. Check gradients, finite
   losses, seam metrics, GPU memory and per-step time. Reduce batch/crop size if
   necessary while preserving full-resolution sources, targets and coordinates.
9. Record the full-run recovery plan: the panorama job runner currently rejects
   resume/fine-tuning. Do not assume the last checkpoint can resume automatically.

## 4. Run stitching training

```bash
python -m panorama.pano_ai.train.run_training \
  --config /runs/stitching_job.yaml
```

1. Capture stdout/stderr and preserve `training_log.jsonl`.
2. Inspect the generated `panorama/training.yaml`. Confirm actual data paths,
   dimensions, seam loss and run settings before continuing the full job.
3. Monitor validation loss, `seam_l1`, `seam_gradient_l1` and `seam_pixels`, along
   with L1/PSNR/SSIM. Zero seam error with zero seam pixels means no seam measured.
4. Preserve best/last checkpoints, effective config and run report.
5. Record the best checkpoint's actual stored contract and selected epoch. The
   current local native RGB contract is v7; verify this in the executed checkout.
6. Compare crop-based results only with equivalent evaluation protocols; changed
   boundary validation crops make older headline metrics an imperfect comparison.

## 5. Run stitching-only inference

1. Create a separate inference config with `ai_pipeline.mode: tiled_neural`,
   `rgb_residual`, matching residual/exposure settings, 12000×6000 model output
   and EMA enabled when available.
2. Disable all correction toggles and keep legacy-checkpoint bypass disabled.
3. Run each evaluation scene using the selected stitching checkpoint:

```bash
python -m panorama.pano_ai.tiled_inference \
  --scene /data/bundle/panorama/val/scene_0001 \
  --config /runs/stitching_inference.yaml \
  --checkpoint /runs/stitching/panorama/panorama_native_detail_best.pt \
  --output-dir /runs/stitching_inference
```

4. Use absolute config/checkpoint paths. The tool appends the scene name to the
   output root.
5. Check actual image dimensions, `metadata.json`, checkpoint contract, EMA use,
   exposure settings, correction status and `coverage_mask.png`.
6. Preserve `initial_panorama.png`; this is the correction-training input, before
   post-stitch corrections.

## 6. Evaluate seams before correction

Run comparisons on representative sky, wires, buildings and ground boundaries:

```bash
python -m panorama.pano_ai.train.diagnose_seams \
  --scene /data/bundle/panorama/val/scene_0001 \
  --config /runs/stitching_inference.yaml \
  --out /runs/seam_checks/scene_0001_sky \
  --region 1500 2000 1024 1024 \
  --prediction /runs/stitching_inference/scene_0001/initial_panorama.png
```

1. Replace the example region with observed seam coordinates. Arguments are
   top, left, height, width on the full output grid. Use a new directory per crop.
2. Compare ownership, photometric, trained and reference crops plus coverage/seam
   masks and JSON metrics. Omit `--prediction` only for source-baseline comparisons.
3. Report tonal discontinuities separately from broken/misaligned structures.
4. Check longitude wrap, fine texture, coverage and calibration alignment.
5. Fix major geometry/calibration problems before correction training. The current
   seam loss does not implement learned geometric flow or repair missing sky.
6. Select and freeze the stitching checkpoint used to create correction inputs.

## 7. Generate correction inputs and review targets

1. Run stitching-only inference on every correction training and validation scene
   using the same frozen stitcher and preprocessing settings.
2. Preserve the site split. Never move a site's variants between splits.
3. Use actual `initial_panorama.png` as `before`, not PTGui `Stitched.jpg`.
4. Use reviewed, clean, geometrically aligned panoramas as `after`.
5. Reject unwanted inserted labels/objects or unrelated edits. Verify projection,
   orientation, matching dimensions and local alignment around seams.
6. Do not resize one image simply to force matching dimensions. If reviewed targets
   are unavailable, report that dependency; do not mark alignment verified falsely.
7. Keep a record of target provenance, exclusions and review decisions.

## 8. Create the local correction manifest and bundle

Example schema; replace paths and include all reviewed train/validation pairs:

```json
{
  "version": 1,
  "pairs": [
    {
      "scene_id": "scene_0001",
      "site": "site_a",
      "split": "train",
      "domain": "drone",
      "projection": "equirectangular",
      "alignment_verified": true,
      "before_source": "ai_stitcher",
      "stitcher_checkpoint": "/runs/stitching/panorama/panorama_native_detail_best.pt",
      "before": "/runs/stitching_inference/scene_0001/initial_panorama.png",
      "after": "/data/reviewed/scene_0001.png"
    }
  ]
}
```

Use safe unique scene IDs and real site identifiers. Both image paths must exist
locally; S3 object keys need downloading first. Keep the original downloaded
manifest; create a new AI-output manifest rather than relabeling PTGui inputs.

```bash
python -m panorama.pano_ai.data.prepare_pairs \
  --manifest /data/ai_correction_pairs.json \
  --out /data/ai_correction_bundle \
  --require-ai-outputs
```

Review the preparation report, train/validation counts, pair provenance and site
separation. The packaged bundle must have `restoration/train` and `restoration/val`.

## 9. Configure and run correction training

1. Copy `panorama/pano_ai/train/configs/ai_output_correction_job.yaml` to a new
   run-specific job file.
2. Set absolute dataset, pipeline-config and new output paths.
3. Keep `tasks: [combined]`. Verify 1024×1024 crops, loss, crop sampling, epochs,
   channels, learning rate, worker count and other supported settings.
4. Run a small pilot in its own output directory; inspect finite losses and memory.
5. Run the full job:

```bash
python -m panorama.pano_ai.train.run_training \
  --config /runs/correction_job.yaml
```

6. Preserve logs, effective settings, best/last checkpoints and validation reports.
7. Compare against the unchanged-input baseline on held-out sites. Inspect seam
   crops and structural detail as well as whole-image metrics.
8. Select the correction checkpoint only after validating its benefit. Record
   scenes made worse. A trained checkpoint alone does not establish improvement.

## 10. Run stitching plus correction inference

1. Create a separate corrected-inference config, retaining the frozen stitching
   checkpoint's architecture, dimensions and exposure settings.
2. Configure the selected combined checkpoint:

```yaml
correction:
  allow_untrained: false
  use_ema: true
  checkpoints:
    combined: /runs/correction/combined/combined_best.pt
  toggles:
    combined: true
    glare: false
    dots: false
    nadir_zenith: false
    color: false
    sharpen: false
```

3. Keep other correction stages disabled unless separately trained and validated.
4. Run the same entry point with the same stitching checkpoint:

```bash
python -m panorama.pano_ai.tiled_inference \
  --scene /data/bundle/panorama/val/scene_0001 \
  --config /runs/corrected_inference.yaml \
  --checkpoint /runs/stitching/panorama/panorama_native_detail_best.pt \
  --output-dir /runs/corrected_inference
```

This command recomputes stitching, then applies combined correction. It does not
take a saved panorama as its CLI input. Keep model weights/settings unchanged so
initial and corrected results are comparable.

5. Inspect metadata to confirm correction executed with the selected checkpoint.
6. Preserve initial/final outputs, masks and metadata for every evaluation scene.

## 11. Final acceptance and handoff

Compare the same held-out scenes and native-scale crops:

```text
Native ownership → photometric base → stitcher-only → corrected → reviewed target
```

Verify:

- Output is actually 12000×6000.
- Tonal seams, wires, building edges, texture and colour improve or are preserved.
- Valid captured regions are not damaged by correction.
- Seam and whole-image metrics are reported with their masks/evaluation scope.
- Failures and scene-specific regressions are documented.
- Missing sky filling is assessed separately; do not claim it from stitching loss.
- The independent test split remains separate from checkpoint selection.

Archive code revision and dirty diff, environment, effective configs, original and
derived manifests, split/site records, checkpoint identities, logs, metrics,
metadata and representative visual comparisons. Report which checks passed,
which limitations remain, and exactly which checkpoints/configs are recommended.

Do not start correction training before generating and reviewing actual AI inputs.
Do not enable correction as the default until held-out evaluation shows benefit.

## Execution record

| Stage | Status | Artifacts / evidence / blockers |
| --- | --- | --- |
| Environment and code verification | Pending | |
| Full-resolution preparation and calibration review | Pending | |
| Stitching pilot | Pending | |
| Stitching training | Pending | |
| Stitching-only inference | Pending | |
| Seam diagnostics and frozen checkpoint selection | Pending | |
| AI-output correction pairs and target review | Pending | |
| Correction packaging | Pending | |
| Correction pilot | Pending | |
| Correction training | Pending | |
| Corrected inference | Pending | |
| Held-out acceptance and archive | Pending | |
