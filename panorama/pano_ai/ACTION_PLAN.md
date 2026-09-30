# Mobile panorama action plan

**First deliverable: geometric stitching followed by one correction model
trained using your PTGui-before / final-after panorama pairs.**

| Priority | Gap | Action | Status |
| --- | --- | --- | --- |
| 1 | No separate labels for each correction model | Train one combined model from before/after pairs | Implemented; synthetic execution tested |
| 2 | Pair alignment and quality are unverified | Review heading, projection, dimensions, crop and edits | Waiting for the dataset; it is not currently available locally |
| 3 | DSLR/drone training differs from mobile inference | Generate mobile-style views; select checkpoints with separate real-mobile validation | Generator and validation implemented and tested; real reviewed data still needed |
| 4 | Finished pairs do not teach source geometry | Retain calibrated geometric stitching; leave learned alignment/blending disabled | Baseline available |
| 5 | Real quality and AWS behavior are unknown | Evaluate on held-out mobile scenes; run GPU/S3 pilot before scaling | Local tests and simulated S3 passed; live AWS pending |

## What changed now

`train/configs/training_job.yaml` selects `tasks: [combined]`. Other trainers
remain available but are not required for this milestone. Combined restoration
predicts an RGB residual without defect masks at training or inference. It can
change any pixel: it does not guarantee preservation of clean regions or reliable
automatic leg removal. Major geometry errors still belong to the stitcher.

The paired loader checks dimensions and orientation and samples native crops,
including the longitude boundary. It requires an explicit alignment review;
equal dimensions alone do not prove alignment. Training and validation must use
different physical-scene IDs, with all variants of a location in one split.
Training crops vary across epochs; validation uses fixed crops for comparison.
Full-scene mobile review is still required before activation.

## Prepare a reviewed pair

```text
training_bundle/restoration/train/scene_001/
  before.png
  after.png
  pair.json
training_bundle/restoration/val/scene_002/
  before.png
  after.png
  pair.json
```

Each `pair.json` identifies the PTGui output and its final corrected version:

```json
{
  "scene_id": "scene_001",
  "before": "before.png",
  "after": "after.png",
  "projection": "equirectangular",
  "alignment_verified": true
}
```

Paths are relative to the manifest; JPEG/TIFF filenames also work. Set
`alignment_verified` only after checking matching heading, crop and coordinates.
This first loader supports full-sphere 2:1 panoramas. Partial panoramas need a
future validity-aware adapter. Images are converted to 8-bit RGB; normalize color
space before pairing. HDR/color-managed ingestion is not implemented.

Set `dataset_root` in `training_job.yaml`, then run:

```powershell
python -m panorama.pano_ai.train.run_training --config panorama/pano_ai/train/configs/training_job.yaml
```

The checkpoint is `<output_dir>/combined/combined_best.pt`. After mobile
validation, point `correction.checkpoints.combined` there and enable
`correction.toggles.combined` in `stitching/config.yaml`. Other post-blend
correction toggles must remain false. Combined inference is disabled for now.

## Acceptance before activating

- Successfully train on 2–3 actual reviewed scene pairs.
- Compare with the uncorrected baseline on separate mobile scenes: floors,
  ceiling lines, blur, longitude join and unwanted changes in clean regions.
- If PTGui-trained repairs do not transfer, add outputs from our mobile stitcher
  paired with aligned corrected targets before enabling the model.

Synthetic execution tests do not establish real quality or convergence.

See [mobile adaptation details in README.md](README.md#mobile-generation-and-validation-details) for the implemented generator and
separate real-mobile validation workflow. PTGui project import and real-data
review remain separate work; the generator starts from reviewed final panoramas.

Local combined-model execution result: `outputs/paired_correction_smoke_verified/smoke_report.json`.
To repeat in a new folder:

```powershell
python -m panorama.pano_ai.train.smoke_paired --out outputs/paired_correction_smoke_new
```

## Latest local verification (2026-09-29)

Combined smoke report regenerated: three synthetic scenes, two CPU epochs, checkpoint reload passed. The 44 AI regression tests passed; 14 contract tests passed again after source-preserving TIFF output was added. See `outputs/plan_verification/report.json`. Real-data quality and AWS validation remain pending.
