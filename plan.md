# Capture360 Product Development Plan

> **Repository:** `Capture_360`
> **Audit basis:** current local checkout (branch history not verified)
> **Product:** Professional 360° panorama capture, AI stitching and correction platform
> **Last updated:** 2026-09-29

## Status legend

- **DONE** - Implementation exists for the stated code scope; not a production-quality claim.
- **PARTIAL** - Foundation exists; integration, validation or acceptance remains.
- **TODO** - No implementation/evidence identified in this checkout; still planned.
- **BLOCKED** - Requires reviewed data, trained weights or an external environment.
- **DEFERRED** - Optional separate-model work outside the current combined-model milestone.
- **SUPERSEDED** - Old architecture or workflow replaced by the current approach.

## Audit scope and current direction (2026-09-29)

This plan combines the source-code audit with subsequent local CPU tests listed
below. Android-device, real-data quality and live AWS validation remain pending.
Checked boxes mean the bounded implementation exists. Unchecked boxes include
remaining work even when a prototype is present. Business and deployment items
without repository evidence remain TODO; their completion outside this checkout
has not been verified. Historical branch-merge claims are not verified here.

**Immediate route:** calibrated geometric stitching - one combined RGB correction
model trained from reviewed PTGui-before / corrected-after pairs - held-out real
mobile evaluation - AWS GPU/S3 pilot. The tiled panorama decoder and separately
labelled correction heads are optional experiments, not prerequisites.

The combined smoke report was regenerated at `outputs/paired_correction_smoke_verified/smoke_report.json`.
The two-epoch CPU run and checkpoint reload passed. The 44-test AI regression
suite passed; 14 contract tests passed after adding TIFF output. After resume
support was added, 12 resume/mobile/paired-data tests passed, including exact CPU
weight and validation-score equivalence with uninterrupted training. These are
overlapping test runs, not 70 distinct tests. No real-data convergence or commercial-quality parity
has been established.

Evidence: [training guide](panorama/pano_ai/README.md),
[architecture](panorama/ARCHITECTURE_VARIABLE_TILED.md),
[training job](panorama/pano_ai/train/configs/training_job.yaml),
[camera contract](panorama/stitching/capture.py),
[AI tests](panorama/pano_ai/tests),
[Android source](app/src/main/java/com/prasanna/capture360).

---

# Overall Roadmap

| Phase | Area | Status | Immediate outcome |
|---|---|---|---|
| 0 | Product & Business Definition | PARTIAL | Freeze MVP business model and success metrics |
| 1 | Panorama Capture Technology | PARTIAL | Reliable Android capture package with poses and camera metadata |
| 2 | Classical Stitching Baseline | PARTIAL | Stable reference stitcher and benchmark |
| 3 | Pano_AI Core | PARTIAL | Train and validate the learned panorama model |
| 4 | Panorama Correction Pipeline | PARTIAL | Combined model implemented; real training and mobile acceptance pending |
| 5 | Python Reference Inference | PARTIAL | Deterministic end-to-end Python inference |
| 6 | AI Model Deployment | PARTIAL | ONNX/TensorRT/cloud inference service |
| 7 | Backend Platform | TODO | APIs for users, agents, jobs and processing |
| 8 | Cloud Storage & Data | PARTIAL | Secure scalable image/panorama storage |
| 9 | Customer Application | TODO | Customer request-to-panorama workflow |
| 10 | Agent Application | PARTIAL | Complete professional agent workflow around capture |
| 11 | Agent Marketplace | TODO | Job assignment and agent dispatch |
| 12 | Payment System | TODO | Customer payment, commission and agent payout |
| 13 | Admin Portal | TODO | Operations, support, reprocessing and monitoring |
| 14 | Security & Privacy | TODO | Production-grade security, privacy and auditability |
| 15 | Quality & Production Testing | PARTIAL | Systematic quality, performance and reliability validation |
| 16 | Pilot Launch | TODO | Controlled real-world pilot |
| 17 | Public Launch | TODO | Production launch and commercial operation |

---

# Phase 0 — Product & Business Definition

**Status: PARTIAL**

## Goal

Define exactly who pays, what service is delivered, how agents participate, and how the business makes money.

## Current decisions

- Phase 1 focuses on **professional/agent capture + AI stitching**.
- Initial customer candidates:
  - Real-estate brokers
  - Shops/business owners
  - Marriage halls/event venues
- A customer requests a panorama service.
- A trained agent performs the capture.
- Images are processed by the Capture360 AI pipeline.
- The finished panorama is delivered to the customer.
- Marketplace/dispatch functionality is deferred until the core service is proven.

## Action items

- [ ] **TODO** - Define customer journey from request to panorama delivery. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Define agent journey from onboarding to completed job. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Define customer profile. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Define agent profile. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Define service area for initial launch. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Define price per panorama/job. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Define agent payout. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Define platform commission. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Define cancellation/refund policy. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Define minimum viable business metrics. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Define MVP scope and explicitly defer non-MVP features. - Requires product/operations decision or implementation and acceptance evidence.

## Exit criteria

The team can answer:
1. Who is the customer?
2. Who performs the capture?
3. What does the customer pay?
4. What does the agent earn?
5. What does Capture360 retain?
6. What is the minimum successful job?

---

# Phase 1 — Panorama Capture Technology

**Status: PARTIAL**

## Goal

Build a reliable Android capture workflow that produces a complete, uploadable capture session.

## Completed / available

- [x] **DONE** - Android project created. - Android foundation exists; device reliability is not established by this audit.
- [x] **DONE** - Camera preview foundation. - Android foundation exists; device reliability is not established by this audit.
- [x] **DONE** - Frame capture foundation. - Android foundation exists; device reliability is not established by this audit.
- [x] **DONE** - Motion sensor integration foundation. - Android foundation exists; device reliability is not established by this audit.
- [x] **DONE** - Orientation estimation foundation. - Android foundation exists; device reliability is not established by this audit.
- [x] **DONE** - Angular coverage tracking foundation. - Android foundation exists; device reliability is not established by this audit.
- [ ] **PARTIAL** - Frame gating foundation. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [x] **DONE** - Guidance overlay foundation. - Android foundation exists; device reliability is not established by this audit.
- [ ] **PARTIAL** - Dual-camera architecture merged into `prasannac10-patch-1`. - Capture profiles exist; branch merge and physical dual-camera behavior not verified.
- [x] **DONE** - Fisheye capture target: 4–8 images. - UI frame-count target exists; capture-quality validation remains.
- [x] **DONE** - Phone/pinhole capture target: 20–30 images. - UI frame-count target exists; capture-quality validation remains.

## Action items

- [ ] **PARTIAL** - Stabilize `CameraController.kt`. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **PARTIAL** - Stabilize `SensorFusionManager.kt`. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **PARTIAL** - Complete `ComplementaryFilter.kt`. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **PARTIAL** - Complete orientation representation and utilities. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **PARTIAL** - Improve IMU drift correction. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [x] **DONE** - Add magnetometer fusion where appropriate. - SensorFusionManager consumes magnetometer and accelerometer data; device calibration still needs validation.
- [ ] **TODO** - Add VIO-lite refinement if required. - Pending; no completion evidence identified in this checkout.
- [ ] **PARTIAL** - Improve angular coverage heatmap. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **PARTIAL** - Finalize frame-gating rules. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **PARTIAL** - Add capture quality validation. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **PARTIAL** - Detect insufficient angular coverage. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **PARTIAL** - Detect excessive motion / poor frames. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **TODO** - Define retry flow for bad captures. - Implement durable capture.json packaging and capture recovery; current poses are held in memory.
- [ ] **TODO** - Capture camera calibration/intrinsic metadata. - App uses an estimated FOV profile; implement calibrated per-frame intrinsic export.
- [ ] **PARTIAL** - Capture camera projection metadata. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **PARTIAL** - Capture per-frame pose metadata. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **TODO** - Package images + poses + camera metadata as one capture session. - Implement durable capture.json packaging and capture recovery; current poses are held in memory.
- [ ] **PARTIAL** - Support offline capture and retryable upload. - Camera/sensor/coverage/gating code exists; device tests and persistent session export remain.
- [ ] **TODO** - Validate true fisheye source/calibration separately from normal phone camera. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Test fisheye workflow with 4, 6 and 8 frames. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Test phone workflow with 20, 24 and 30 frames. - Pending; no completion evidence identified in this checkout.

## Important constraint

Selecting a "fisheye" mode in the UI does not turn a normal phone camera into a physical fisheye camera. A true fisheye workflow requires an appropriate camera source and calibration.

## Exit criteria

An Android capture session can reliably produce:

```text
images/
poses.json or poses.pt
camera.json
capture metadata
quality information
```

---

# Phase 2 — Classical Stitching Baseline

**Status: PARTIAL**

## Goal

Maintain a classical stitcher as a reference/baseline for comparison with Pano_AI.

## Current state

- [x] **DONE** - OpenCV-based stitching foundation exists. - Classical implementation exists in panorama/pano_classical; benchmark quality remains unverified.
- [x] **DONE** - Classical stitching remains separate from the learned architecture. - Classical implementation exists in panorama/pano_classical; benchmark quality remains unverified.
- [x] **DONE** - Fisheye stitching prototype exists. - Classical implementation exists in panorama/pano_classical; benchmark quality remains unverified.

## Action items

- [x] **DONE** - **NEW:** Overlap detection with ORB/SIFT/AKAZE feature matching - Classical implementation exists in panorama/pano_classical; benchmark quality remains unverified.
- [x] **DONE** - **NEW:** Seam detection using minimum-cost path DP - Classical implementation exists in panorama/pano_classical; benchmark quality remains unverified.
- [x] **DONE** - **NEW:** Multi-band Laplacian blending, Feather blending, Graph-Cut blending - Classical implementation exists in panorama/pano_classical; benchmark quality remains unverified.
- [x] **DONE** - **NEW:** Ghost removal with Poisson inpainting and median filtering - Classical implementation exists in panorama/pano_classical; benchmark quality remains unverified.
- [x] **DONE** - **NEW:** Parallax correction from stereo matching and feature-based flow - Classical implementation exists in panorama/pano_classical; benchmark quality remains unverified.
- [ ] **TODO** - Stabilize classical fisheye stitching. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Add/validate phone/pinhole baseline. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Validate camera calibration handling. - Pending; no completion evidence identified in this checkout.
- [ ] **BLOCKED** - Compare 4/6/8 fisheye frames. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Compare 20/24/30 phone frames. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Measure seam quality. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Measure exposure consistency. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Evaluate moving objects. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Evaluate repetitive textures. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Evaluate poles/nadir/zenith. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Measure processing time. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Build a fixed classical-vs-AI benchmark dataset. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Store baseline outputs for every benchmark scene. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.

## Exit criteria

A repeatable baseline exists against which every Pano_AI model version can be compared.

---

# Phase 3 — Pano_AI Core

**Status: PARTIAL**

## Goal

Maintain the experimental tiled decoder; prioritize source-preserving geometry plus combined correction for the available paired dataset.

## Historical decoder architecture (superseded)

See [current block diagrams](panorama/ARCHITECTURE_VARIABLE_TILED.md). Names below describe the old design, not the recommended training route.

```text
Input images
    ↓
ImageEncoder
    ↓
SetAggregator
    ↓
SphericalFusion
    ↓
PanoramaDecoder
    ↓
Initial panorama
```

The model must support:
- Fisheye cameras: approximately 4–8 images.
- Normal phone/pinhole cameras: approximately 20–30 images.
- Variable number of input images.
- Frame masking/padding.
- Camera-aware geometry.
- Per-frame poses.

## Completed

- [x] **DONE** - Configurable image encoder. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.
- [x] **DONE** - ResNet18 encoder option. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.
- [x] **DONE** - Optional pretrained encoder. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.
- [x] **DONE** - Spatial feature-map retention. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.
- [x] **DONE** - Feature projection. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.
- [ ] **SUPERSEDED** - SetAggregator. - Current model uses TileAttentionAggregator and project_tile_features; see architecture document.
- [x] **DONE** - Variable-N input handling. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.
- [x] **DONE** - Frame mask handling. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.
- [ ] **SUPERSEDED** - SphericalFusion. - Current model uses TileAttentionAggregator and project_tile_features; see architecture document.
- [x] **DONE** - Camera-aware projection foundation. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.
- [x] **DONE** - Panorama decoder. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.
- [x] **DONE** - End-to-end PanoramaModel structure. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.
- [x] **DONE** - EMA support. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.
- [x] **DONE** - YAML-driven configuration foundation. - Model/training foundation exists; see panorama/pano_ai/models and utils; real trained quality is pending.

## Partial / pending

- [ ] **TODO** - Fully validate fisheye projection. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Fully validate pinhole projection. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Validate mixed camera metadata handling. - Pending; no completion evidence identified in this checkout.
- [ ] **DEFERRED** - Finalize geometry loss. - Geometry loss is not implemented; nonzero unsupported weights fail explicitly in panorama_loss.py.
- [x] **DONE** - Finalize supervised L1/SSIM loss configuration. - panorama_loss.py implements explicit L1/SSIM/perceptual weights.
- [x] **DONE** - Validate checkpoint save/load workflow. - Checkpoint contracts are implemented; combined smoke training and checkpoint reload passed on CPU.
- [x] **DONE** - Validate EMA checkpoint inference. - Explicit EMA selection is implemented; the AI regression suite passed. The broader legacy smoke report has not been regenerated.
- [ ] **BLOCKED** - Validate 14K dataset end-to-end. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Train first full model on the actual dataset. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Evaluate quantitatively. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Evaluate visually. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Compare against classical baseline. - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **TODO** - Analyze failures by camera type and frame count. - Pending; no completion evidence identified in this checkout.

## Dataset requirements

Recommended organization:

```text
dataset/
  scenes/
    scene_000001/
      images/
      poses.pt
      camera.json
      panorama.png
```

## Dataset action items

- [x] **DONE** - Finalize scene organization. - Shared capture schema and paired-scene contract implemented; actual dataset population/review remains.
- [x] **DONE** - Finalize camera metadata. - Shared capture schema and paired-scene contract implemented; actual dataset population/review remains.
- [x] **DONE** - Finalize pose metadata. - Shared capture schema and paired-scene contract implemented; actual dataset population/review remains.
- [x] **DONE** - Split by capture scene/session, not individual image. - Shared capture schema and paired-scene contract implemented; actual dataset population/review remains.
- [ ] **TODO** - Create 80/10/10 train/validation/test split. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Stratify by camera type. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Stratify by frame count. - Pending; no completion evidence identified in this checkout.
- [ ] **PARTIAL** - Validate all image/pose/camera relationships. - Structural validators exist; audit the complete real dataset when available.
- [ ] **TODO** - Create dataset visualization tools. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Build evaluation buckets: - Pending; no completion evidence identified in this checkout.
  - [ ] **TODO** - Fisheye 4–6 frames. - Pending; no completion evidence identified in this checkout.
  - [ ] **TODO** - Fisheye 7–8 frames. - Pending; no completion evidence identified in this checkout.
  - [ ] **TODO** - Phone 20–24 frames. - Pending; no completion evidence identified in this checkout.
  - [ ] **TODO** - Phone 25–30 frames. - Pending; no completion evidence identified in this checkout.
  - [ ] **TODO** - Wrap-around regions. - Pending; no completion evidence identified in this checkout.
  - [ ] **TODO** - Pole/nadir/zenith regions. - Pending; no completion evidence identified in this checkout.

## Exit criteria

A trained checkpoint produces a usable initial panorama on held-out scenes and demonstrates measurable improvement versus the classical baseline.

---

# Phase 4 — Panorama Correction Pipeline

**Status: PARTIAL**

## Goal

Train one combined correction model from reviewed panorama pairs. Keep separate-stage models optional until their labels and validated weights exist.

## Completed

### Baseline Correction Stages (Learned)
- [x] **DONE** - Glare U-Net architecture - Architecture/helper code exists; this does not establish trained weights or production integration.
- [x] **DONE** - Lens-dot classical processing - Architecture/helper code exists; this does not establish trained weights or production integration.
- [x] **DONE** - Nadir/Zenith mask-aware U-Net - Architecture/helper code exists; this does not establish trained weights or production integration.
- [x] **DONE** - Color enhancement U-Net - Architecture/helper code exists; this does not establish trained weights or production integration.
- [x] **DONE** - Classical sharpening - Architecture/helper code exists; this does not establish trained weights or production integration.

### Advanced Correction Models (NEW - Both Classical & Learned)
- [x] **DONE** - **Overlap Detection** - Architecture/helper code exists; this does not establish trained weights or production integration.
  - [x] **DONE** - Classical: `OverlapDetectorCV` (ORB/SIFT/AKAZE, RANSAC homography) - Architecture/helper code exists; this does not establish trained weights or production integration.
  - [x] **DONE** - Learned OverlapDetectionUNet class exists in advanced_corrections.py; no validated production weights or post-blend integration.

- [x] **DONE** - **Parallax Correction** - Architecture/helper code exists; this does not establish trained weights or production integration.
  - [x] **DONE** - Classical: `ParallaxCorrectionCV` (Stereo SGBM, RBF interpolation) - Architecture/helper code exists; this does not establish trained weights or production integration.
  - [x] **DONE** - Learned: `ParallaxCorrectionUNet` (U-Net residual learning) - Architecture/helper code exists; this does not establish trained weights or production integration.

- [x] **DONE** - **Seam-Aware Blending** - Architecture/helper code exists; this does not establish trained weights or production integration.
  - [x] **DONE** - Classical: `SeamBlendingCV` (Multiband Laplacian, Feather, Graph-Cut) - Architecture/helper code exists; this does not establish trained weights or production integration.
  - [x] **DONE** - Learned: `SeamBlendingUNet` (Pixel-wise blend weight learning) - Architecture/helper code exists; this does not establish trained weights or production integration.

- [x] **DONE** - **Ghost Artifact Removal** - Architecture/helper code exists; this does not establish trained weights or production integration.
  - [x] **DONE** - Classical: `GhostRemovalCV` (Poisson inpainting, median filter, optical flow) - Architecture/helper code exists; this does not establish trained weights or production integration.
  - [x] **DONE** - Learned: `GhostRemovalUNet` (Artifact inpainting U-Net) - Architecture/helper code exists; this does not establish trained weights or production integration.

- [x] **DONE** - **Parallax Flow Detection** - Architecture/helper code exists; this does not establish trained weights or production integration.
  - [x] **DONE** - Learned: `ParallaxCorrectionDetector` (2D flow field prediction) - Architecture/helper code exists; this does not establish trained weights or production integration.

### Historical pipeline architecture

The numbered four-stage sequence below is historical. Current inference rejects unsupported pairwise post-blend stages; combined restoration is enabled alone.
- [ ] **SUPERSEDED** - 4-stage correction pipeline: - Current dispatcher, source_inference and HighResolutionCorrectionPipeline replace the historical orchestration.
  1. Stage 1: Baseline learned corrections (glare, nadir_zenith, color)
  2. Stage 2: Advanced geometric corrections (parallax, ghost removal, overlap validation)
  3. Stage 3: Seam-aware blending
  4. Stage 4: Classical finishing (dots, sharpening)

- [ ] **SUPERSEDED** - Dual-path factory pattern (`DualPathPanoramaPipeline`) - Current dispatcher, source_inference and HighResolutionCorrectionPipeline replace the historical orchestration.
  - Switch between classical and learned methods via config
  - Factory pattern for clean method selection
  - Unified orchestration layer

- [x] **DONE** - Configuration system - Architecture/helper code exists; this does not establish trained weights or production integration.
  - `advanced_corrections` section with per-model toggles
  - `classical_methods` boolean flags
  - `classical_parameters` for OpenCV tuning

### Integration
- [ ] **SUPERSEDED** - Updated `pipeline.py` with 4-stage architecture - Current dispatcher, source_inference and HighResolutionCorrectionPipeline replace the historical orchestration.
- [x] **DONE** - Five model classes exist in advanced_corrections.py; real training and supported pipeline integration remain separate requirements.
- [ ] **SUPERSEDED** - Overlap detector AI in `overlap_detector.py` - The historical standalone file is absent; OverlapDetectionUNet exists in advanced_corrections.py. No new duplicate file is needed.
- [ ] **SUPERSEDED** - Classical methods in `stitching/` folder - Classical implementations now live in panorama/pano_classical; shared contracts live in stitching.
- [x] **DONE** - Updated `config.yaml` with all toggles - Architecture/helper code exists; this does not establish trained weights or production integration.
- [ ] **SUPERSEDED** - Unified `dual_path_pipeline.py` orchestration - Current dispatcher, source_inference and HighResolutionCorrectionPipeline replace the historical orchestration.

### Documentation
- [ ] **PARTIAL** - IMPLEMENTATION_SUMMARY.md exists but requires a current-code audit; historical completion claims are not acceptance evidence.
  - Architecture diagrams
  - File structure overview
  - Model specifications
  - Usage examples
  - Training workflow
  - Classical vs Learned comparison table

## Pending (After Training)

- [ ] **DEFERRED** - Prepare parallax training data (depth maps or flow ground truth) - Separate-stage labels/weights are optional; first train combined on reviewed before/after pairs.
- [ ] **DEFERRED** - Prepare ghost artifact training data - Separate-stage labels/weights are optional; first train combined on reviewed before/after pairs.
- [ ] **DEFERRED** - Prepare seam blending training data - Separate-stage labels/weights are optional; first train combined on reviewed before/after pairs.
- [ ] **DEFERRED** - Prepare overlap detection training data - Separate-stage labels/weights are optional; first train combined on reviewed before/after pairs.
- [ ] **DEFERRED** - Train ParallaxCorrectionUNet - Separate-stage labels/weights are optional; first train combined on reviewed before/after pairs.
- [ ] **DEFERRED** - Train GhostRemovalUNet - Separate-stage labels/weights are optional; first train combined on reviewed before/after pairs.
- [ ] **DEFERRED** - Train SeamBlendingUNet - Separate-stage labels/weights are optional; first train combined on reviewed before/after pairs.
- [ ] **DEFERRED** - Train OverlapDetectionUNet - Class exists in advanced_corrections.py; labelled training, validated weights and deployment remain.
- [ ] **DEFERRED** - Train ParallaxCorrectionDetector - Separate-stage labels/weights are optional; first train combined on reviewed before/after pairs.
- [ ] **BLOCKED** - Evaluate each model independently - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Benchmark classical methods baseline - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Compare classical vs learned quality - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Compare classical vs learned speed - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **TODO** - Enable trained models in config.yaml - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Integrate into production pipeline - Pending; no completion evidence identified in this checkout.

## Important rule

The correction models are independent stages. Each must be trained and evaluated independently before being enabled in production.

## Exit criteria

✅ **IMPLEMENTED:** Complete dual-path correction pipeline with 5 new AI models and 4 classical method implementations.

**NEXT MILESTONE:** Train individual AI models and validate against classical baselines.

---

# Phase 5 — Python Reference Inference

**Status: PARTIAL**

## Goal

Create the authoritative Python reference path before deployment to mobile/cloud inference.

## Historical target flow (superseded)

Current default: calibrated sources -> geometric composition -> optional combined correction -> final panorama and metadata.

```text
Raw images
    ↓
poses + camera metadata
    ↓
PanoramaModel
    ↓
Initial panorama
    ↓
CorrectionPipeline (now dual-path!)
    ↓
Final panorama
```

## Current state

- [x] **DONE** - Dataset/DataLoader foundation. - Current Python inference foundation exists; real-scene acceptance remains pending.
- [x] **DONE** - Variable-N collation. - Current Python inference foundation exists; real-scene acceptance remains pending.
- [x] **DONE** - Pose normalization. - Current Python inference foundation exists; real-scene acceptance remains pending.
- [x] **DONE** - Camera parameter collation. - Current Python inference foundation exists; real-scene acceptance remains pending.
- [x] **DONE** - Panorama model inference foundation. - Current Python inference foundation exists; real-scene acceptance remains pending.
- [x] **DONE** - Checkpoint loading. - Current Python inference foundation exists; real-scene acceptance remains pending.
- [x] **DONE** - EMA support. - Current Python inference foundation exists; real-scene acceptance remains pending.
- [ ] **TODO** - Intermediate spherical feature output option. - No current inference feature-export option confirmed; implement only if needed for diagnostics.
- [x] **DONE** - Panorama metadata output foundation. - Current Python inference foundation exists; real-scene acceptance remains pending.
- [ ] **SUPERSEDED** - **NEW:** Integration with dual-path correction pipeline - Current dispatcher, source_inference and HighResolutionCorrectionPipeline replace the historical orchestration.
- [x] **DONE** - **NEW:** Configurable method selection (classical vs learned) - Current Python inference foundation exists; real-scene acceptance remains pending.
- [ ] **PARTIAL** - **NEW:** Auxiliary data handling (seam edges, overlap masks, etc.) - Semantic masks supported; unsupported post-blend pairwise stages raise explicitly.

## Action items

- [x] **DONE** - Ensure `camera_params` are always passed to PanoramaModel. - Current Python inference foundation exists; real-scene acceptance remains pending.
- [x] **DONE** - Complete learned model → correction pipeline integration. - Current Python inference foundation exists; real-scene acceptance remains pending.
- [x] **DONE** - Define correction-mask input/output contract. - stitching/defects.py defines categories, geometry coverage and exact-size masks.
- [ ] **TODO** - Make inference deterministic. - Pending; no completion evidence identified in this checkout.
- [x] **DONE** - Save initial panorama. - Both Python inference routes save the initial panorama.
- [x] **DONE** - Save final panorama. - Both Python inference routes save the final panorama.
- [ ] **PARTIAL** - Save yaw/pitch metadata. - Canonical rotations are exported; a dedicated yaw/pitch output is not standardized.
- [x] **DONE** - Save camera metadata. - source_inference exports capture.json and metadata.json.
- [x] **DONE** - Support TIFF output. - Both Python routes write TIFF; source-preserving initial/final TIFF pixels match PNG in the contract test.
- [ ] **TODO** - Add EXR/HDR output where required. - Pending; no completion evidence identified in this checkout.
- [x] **DONE** - Add automated end-to-end inference test. - Both smoke runners exist; the combined smoke run and checkpoint reload passed. Broader legacy smoke rerun remains separate.
- [x] **DONE** - Add malformed-input validation. - Camera, profiles, pair loaders and checkpoint validators reject invalid inputs.
- [ ] **BLOCKED** - Benchmark CPU inference (classical methods). - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **BLOCKED** - Benchmark GPU inference (learned models). - Needs reviewed benchmark scenes and retained real-run results; synthetic tests are insufficient.
- [ ] **TODO** - Record memory consumption. - Pending; no completion evidence identified in this checkout.
- [x] **DONE** - Document method switching instructions. - Current modes and correction toggles are documented in the AI README.
- [ ] **PARTIAL** - Create inference comparison scripts (classical vs learned). - visual_regression.py compares candidates, references and masks; curated benchmark automation remains.

## Exit criteria

One command can take a complete capture session and produce a final panorama plus metadata reproducibly using either classical or learned methods.

---

# Phase 6 — AI Model Deployment

**Status: PARTIAL**

## Goal

Move the validated Python model into a production inference environment.

## Model export

- [ ] **TODO** - Export PanoramaModel to ONNX. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - **NEW:** Export ParallaxCorrectionUNet to ONNX. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - **NEW:** Export GhostRemovalUNet to ONNX. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - **NEW:** Export SeamBlendingUNet to ONNX. - Pending; no completion evidence identified in this checkout.
- [ ] **DEFERRED** - Export OverlapDetectionUNet - Class exists in advanced_corrections.py; labelled training, validated weights and deployment remain.
- [ ] **TODO** - Validate ONNX output against PyTorch. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Investigate dynamic input-N support. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Validate camera metadata inputs. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Validate output equivalence. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - TensorRT experimentation. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - FP16 inference. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - INT8 experimentation. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Model size optimization. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Latency benchmark. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - GPU memory benchmark. - Pending; no completion evidence identified in this checkout.

## Cloud inference

- [x] **DONE** - Select cloud provider. - AWS chosen for the planned pilot; live deployment remains pending.
- [ ] **TODO** - Define GPU instance/service. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Containerize inference. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Create inference API/service. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Add job queue. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Add worker process. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Add retries. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Add autoscaling. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Add monitoring. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Add failure reporting. - Pending; no completion evidence identified in this checkout.

## Exit criteria

A production-like service can accept a capture session and return a final panorama with predictable latency and cost.

---

# Phase 7 — Backend Platform

**Status: TODO**

## Goal

Provide the business APIs connecting customers, agents, capture sessions and AI processing.

## Action items

- [ ] **TODO** - Select backend technology. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Design REST/API contract. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Implement authentication. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Implement authorization. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - User management. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Agent management. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Customer management. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Job management. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Capture-session management. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Image-set management. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Processing-job management. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Panorama management. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Processing status API. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Error-management API. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Notification integration. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Admin APIs. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - API logging and monitoring. - Requires product/operations decision or implementation and acceptance evidence.

## Core entities

```text
User
Agent
Customer
Job
CaptureSession
ImageSet
ProcessingJob
Panorama
Payment
AgentPayout
```

## Exit criteria

A complete job can move through the backend state machine from request to delivered panorama.

---

# Phase 8 — Cloud Storage & Data

**Status: PARTIAL**

## Goal

Securely store originals, intermediate data and final panoramas at scale.

## Action items

- [x] **DONE** - Select object-storage provider. - S3 chosen for training input; production storage setup remains pending.
- [ ] **PARTIAL** - Define storage hierarchy. - Training bundle and S3 cache layout implemented; production retention/storage layout remains.
- [ ] **TODO** - Upload original images. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Store capture metadata. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Store poses. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Store intermediate processing data where required. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Store final panoramas. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Configure lifecycle policies. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Configure backups. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Define deletion policy. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Add CDN for panorama delivery. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Implement signed/private URLs. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Define storage cost controls. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Monitor storage growth. - Pending; no completion evidence identified in this checkout.

## Exit criteria

A panorama remains securely accessible to the authorized customer while raw data and temporary processing data follow defined retention rules.

---

# Phase 9 — Customer Application

**Status: TODO**

## Goal

Allow customers to request, pay for and consume panorama services.

## Action items

- [ ] **TODO** - Registration/login. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Customer profile. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Create panorama request. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Location capture. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Select service. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Display price. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Payment. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Track request. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Show assigned agent. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Show capture status. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Show processing status. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - View panorama. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Download panorama. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Share panorama. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Job history. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Invoice/receipt. - Requires product/operations decision or implementation and acceptance evidence.

## Exit criteria

A customer can complete the entire request → payment → panorama delivery journey.

---

# Phase 10 — Agent Application

**Status: PARTIAL**

## Goal

Provide professional agents with everything required to perform and complete capture jobs.

## Already available

- [x] **DONE** - Capture application foundation. - Android foundation exists; device reliability is not established by this audit.
- [x] **DONE** - Camera capture. - Android foundation exists; device reliability is not established by this audit.
- [x] **DONE** - Motion/orientation foundation. - Android foundation exists; device reliability is not established by this audit.
- [x] **DONE** - Angular coverage guidance. - Android foundation exists; device reliability is not established by this audit.
- [ ] **PARTIAL** - Dual-camera architecture. - Capture profiles exist; branch merge and physical dual-camera behavior not verified.
- [x] **DONE** - Fisheye capture target: 4–8 images. - UI frame-count target exists; capture-quality validation remains.
- [x] **DONE** - Phone capture target: 20–30 images. - UI frame-count target exists; capture-quality validation remains.

## Agent onboarding

- [ ] **TODO** - Agent registration. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Agent profile. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Phone verification. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Service-area selection. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Availability. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Training content. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Capture-quality certification. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Admin approval. - Pending; no completion evidence identified in this checkout.

## Job workflow

- [ ] **TODO** - Job list. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Job details. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Accept/reject job. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Navigation. - Pending; no completion evidence identified in this checkout.
- [ ] **PARTIAL** - Start capture. - Local capture/viewer foundation exists; job-integrated agent workflow remains.
- [ ] **PARTIAL** - Validate capture quality. - Local capture/viewer foundation exists; job-integrated agent workflow remains.
- [ ] **TODO** - Upload capture session. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Show upload progress. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Show processing status. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Complete job. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Show earnings. - Pending; no completion evidence identified in this checkout.

## Exit criteria

A trained agent can accept a job, capture a valid session, upload it, and receive completion confirmation.

---

# Phase 11 — Agent Marketplace

**Status: TODO**

## Goal

Automate assignment of customer jobs to suitable agents.

## Action items

- [ ] **TODO** - Agent availability. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Agent location. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Nearby-agent search. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Manual assignment. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Automatic assignment. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Job acceptance timeout. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Reassignment. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Ratings. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Cancellation. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Dispute handling. - Requires product/operations decision or implementation and acceptance evidence.

## MVP recommendation

For the first 10–50 real jobs, use **manual/admin assignment** rather than building the complete marketplace.

## Exit criteria

The system can reliably assign jobs to available qualified agents without manual intervention.

---

# Phase 12 — Payment System

**Status: TODO**

## Goal

Handle customer payments and agent payouts reliably.

## Action items

- [ ] **TODO** - Select payment gateway. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Customer payment initiation. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Payment status verification. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Failed-payment handling. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Refund workflow. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Cancellation charges. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Platform commission calculation. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Agent payout calculation. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Payout processing. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Reconciliation. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Invoice generation. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Payment audit trail. - Requires product/operations decision or implementation and acceptance evidence.

## Exit criteria

Every completed job has a traceable financial record:

```text
Customer payment
      ↓
Platform commission
      ↓
Agent payout
      ↓
Reconciliation
```

---

# Phase 13 — Admin Portal

**Status: TODO**

## Goal

Provide operations and support teams with centralized control.

## Action items

- [ ] **TODO** - Admin login. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Customer management. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Agent management. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Agent approval. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Job management. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Manual agent assignment. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Capture-session inspection. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Upload inspection. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Panorama viewing. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Reprocess failed jobs. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Failure handling. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Payment monitoring. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Payout monitoring. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Customer support tools. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Operational metrics. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - AI processing metrics. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Cost monitoring. - Requires product/operations decision or implementation and acceptance evidence.

## Exit criteria

Operations can manage a customer job without engineering intervention.

---

# Phase 14 — Security & Privacy

**Status: TODO**

## Goal

Make customer images, accounts and payments secure enough for production use.

## Action items

- [ ] **TODO** - Authentication. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Role-based access control. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Secure APIs. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - TLS for data in transit. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Encryption at rest. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Private image storage. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Signed URLs. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Data retention policy. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Data deletion workflow. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Audit logs. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Backup and recovery. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Privacy policy. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Terms of service. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Agent agreement. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Payment compliance review. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Security testing. - Requires product/operations decision or implementation and acceptance evidence.

## Exit criteria

Access to customer data is explicitly controlled, auditable and revocable.

---

# Phase 15 — Quality & Production Testing

**Status: PARTIAL**

## Goal

Establish objective quality gates for both AI output and the complete product.

## Panorama test matrix

### Camera / frame count

- [ ] **TODO** - Fisheye — 4 frames. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Fisheye — 6 frames. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Fisheye — 8 frames. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Phone — 20 frames. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Phone — 24 frames. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Phone — 30 frames. - Pending; no completion evidence identified in this checkout.

### Scene conditions

- [ ] **TODO** - Bright indoor. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Dark indoor. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Outdoor. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Reflective surfaces. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Moving people. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Furniture/windows. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Repetitive textures. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Low light. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Glare. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Lens dots. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Nadir/floor. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Zenith/ceiling. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Exposure differences. - Pending; no completion evidence identified in this checkout.

## Product testing

- [ ] **TODO** - Android unit tests. - Pending; no completion evidence identified in this checkout.
- [x] **DONE** - AI unit tests. - 44 AI regression tests passed; subsequent focused runs passed 14 TIFF/contract tests and 12 resume/mobile/paired-data tests.
- [ ] **TODO** - Backend unit tests. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - API tests. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - End-to-end tests. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Load tests. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Security tests. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Payment tests. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Device compatibility tests. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Network interruption tests. - Pending; no completion evidence identified in this checkout.
- [ ] **TODO** - Offline/retry tests. - Pending; no completion evidence identified in this checkout.

## Metrics

Track at minimum:
- Capture success rate.
- Stitch success rate.
- Reprocessing rate.
- Panorama quality score.
- Processing time.
- Upload failure rate.
- Customer satisfaction.
- Agent satisfaction.
- Cost per panorama.
- Revenue per panorama.
- Agent earnings per job.

## Exit criteria

Quality and reliability are measured, repeatable and have explicit release thresholds.

---

# Phase 16 — Pilot Launch

**Status: TODO**

## Goal

Validate the complete business with real customers and real capture conditions.

## Pilot sequence

```text
5 agents
   ↓
10–20 customers
   ↓
50 jobs
   ↓
100 panoramas
   ↓
Analyze failures
   ↓
Improve product
   ↓
500 jobs
```

## Action items

- [ ] **TODO** - Recruit/train first agents. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Select pilot customers. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Define pilot pricing. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Run real capture sessions. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Monitor upload reliability. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Monitor AI processing. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Collect panorama quality feedback. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Collect customer feedback. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Collect agent feedback. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Measure operating cost. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Identify top failure modes. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Fix highest-impact failures. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Repeat pilot. - Requires product/operations decision or implementation and acceptance evidence.

## Exit criteria

The service can repeatedly deliver acceptable panoramas to paying customers at a sustainable operating cost.

---

# Phase 17 — Public Launch

**Status: TODO**

## Goal

Move from controlled pilot to production commercial operation.

## Technical readiness

- [ ] **TODO** - Production cloud environment. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Production database. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Production object storage. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Production AI inference. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Monitoring. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Alerts. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Backups. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Disaster recovery. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - CI/CD. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Android production release. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - iOS production release. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Admin portal production release. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Production security validation. - Requires product/operations decision or implementation and acceptance evidence.

## Business readiness

- [ ] **TODO** - Final pricing. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Payment system. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Customer contracts where required. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Terms of service. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Privacy policy. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Refund policy. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Customer support process. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Agent recruitment process. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Agent training. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Marketing website. - Requires product/operations decision or implementation and acceptance evidence.
- [ ] **TODO** - Customer acquisition process. - Requires product/operations decision or implementation and acceptance evidence.

## Exit criteria

Capture360 can accept customers, dispatch/coordinate agents, process panoramas, collect payment, pay agents and support failures without depending on manual engineering intervention.

---

# Recent verified changes (2026-09-29)

| Change | Status | Evidence / scope |
| --- | --- | --- |
| Restoration checkpoint resume | DONE | `train_restoration.py`, `run_training.py`, `tests/test_resume.py`; 12 focused tests passed |
| Resume validation | DONE | Rejects incompatible task/contract/channels/selection and exhausted epoch targets; CPU equivalence tested |
| Latest-epoch recovery | DONE | Atomic checkpoint replacement; latest embeds earlier best, supporting recovery to a new output folder |
| Lossless TIFF output | DONE | Initial/final source-preserving TIFF matches PNG pixels; 14 contract tests passed |
| Dependency consolidation | DONE | Shared `requirements.txt`, training include plus boto3, optional export/HDR/metrics file |
| Obsolete workflow removal | DONE | Removed `panorama/steps_variable_tiled.txt`; current guide is `panorama/pano_ai/README.md` |
| Generated output exclusion | DONE | Root `.gitignore` contains `/outputs/`; verified with `git check-ignore` |
| Combined training execution | DONE | Three synthetic scenes, two CPU epochs, checkpoint reload; report under `outputs/paired_correction_smoke_verified/` |

Resume is available for combined and other restoration tasks only. It restores
optimizer state (including its saved learning rate), weights, completed epoch and
available RNG state. Configure `combined.resume` with a trusted local checkpoint;
`epochs` is the total desired count. Keep dataset, splits and sampling settings
unchanged. Resume starts at the chosen checkpoint epoch. Restoration now saves *_last.pt at every completed epoch and preserves the earlier best checkpoint inside it.
Old checkpoints without RNG state cannot reproduce the original sampling sequence.
Pairwise/decoder resume and checkpoint upload
remain pending. Cross-device reproducibility has not been established.

Local reports and checkpoints under `outputs/` are intentionally ignored by Git.
Archive them separately when sharing validation evidence or moving to AWS.
`outputs/plan_verification/report.json` records the earlier 44-test and TIFF runs;
the later 12-test resume result is recorded here and in the test-run output.

# Recommended Immediate Milestones

## M1 - Reproduce the local training execution test (P0)

- [x] **DONE** - Combined model, paired loader, preparation CLI and smoke runner implemented.
- [x] **DONE** - Mobile-style generator and separate real-mobile checkpoint selection implemented.
- [x] **DONE** - Regenerated combined smoke report: 3 scenes, 2 epochs, validation L1 0.199613, parameter change 0.120620, checkpoint reload passed. Broader legacy/mobile smoke reports can be regenerated when those paths change.
- [x] **DONE** - 44 AI regression tests passed on the CPU environment; 14 contract tests passed again after TIFF changes. See outputs/plan_verification/report.json for those runs; the later 12-test resume/mobile/paired-data run also passed.

## M2 - Prepare real data and prove mobile quality (P0)

- [ ] **BLOCKED** - Obtain 2-3 reviewed PTGui-before / corrected-after pairs; dataset not currently available locally.
- [ ] **BLOCKED** - Review projection, heading, dimensions, color space and pixel alignment before packaging.
- [x] **DONE** - Physical-scene split leakage checks and automated pair packaging implemented.
- [ ] **BLOCKED** - Populate scene-disjoint train, validation and independent test sets; stratify camera/scene types.
- [ ] **BLOCKED** - Train combined on actual reviewed pairs and inspect full-panorama outputs.
- [ ] **BLOCKED** - Evaluate real mobile stitcher outputs against aligned corrected targets and the uncorrected baseline.
- [ ] **TODO** - Set measured acceptance thresholds for geometry, texture, join continuity, unwanted edits, time and memory.
- [ ] **BLOCKED** - Enable combined correction alone only after those quality gates pass.

Separate displacement/ownership/defect-labelled models remain deferred. The combined
model does not guarantee leg removal or preservation of clean pixels. Geometry
errors should be addressed in calibrated stitching, not assumed repairable by RGB training.

## M3 - Complete capture export and reliability (P0)

- [x] **DONE** - Python shared camera contract and supported pose adapters implemented.
- [ ] **TODO** - Export calibrated per-frame metadata and persistent capture.json from the Android app.
- [ ] **PARTIAL** - Capture, sensor fusion, gating and coverage foundations exist; validate drift, image quality and recovery on devices.
- [ ] **TODO** - Add durable offline session recovery and retryable upload.
- [ ] **TODO** - Validate physical fisheye and phone frame-count/scene-condition buckets.

## M4 - AWS pilot and processing service (P1)

- [x] **DONE** - Local/S3 training input adapter and configuration-driven runner implemented.
- [ ] **TODO** - Run live S3/GPU pilot with IAM, CUDA environment, disk and memory measurements.
- [x] **DONE** - Restoration checkpoint resume restores model, optimizer, epoch and available RNG state; CPU interrupted/uninterrupted equivalence tested. Resumes from the chosen best or latest checkpoint; latest embeds the earlier best.
- [ ] **TODO** - Panorama/pairwise resume and checkpoint output upload remain separate work.
- [ ] **TODO** - Build Python inference container, service, queue, workers and retries.
- [ ] **DEFERRED** - Export/optimize ONNX or TensorRT after freezing and validating the Python runtime contract.

The root Dockerfile builds Android; it is not an AI inference container.
`panorama/pano_ai/export.py` writes a runtime description, not an ONNX model.
The S3 adapter downloads training inputs; it does not implement customer uploads,
signed URLs, production retention, a processing service or checkpoint uploads.

## M5 - Minimum business platform (P1)

- [ ] **TODO** - Customer/agent accounts and role-based access.
- [ ] **TODO** - Job creation, manual assignment and processing status.
- [ ] **TODO** - Authorized panorama delivery, payment and support workflows.
- [ ] **TODO** - Complete security, privacy, monitoring and pilot acceptance from phases 7-17.

# Audit corrections and remaining limitations

The previous "Phase 4 complete" and "train five models next" conclusions are
superseded. Model classes and classical helpers do not establish a complete,
trained correction system. Current code uses the shared dispatcher and explicit
checkpoint/mask contracts; old Pano_AI paths and dual-path factory references
must not be used as current entry points.

Current additions not reflected in the original plan:

- [x] **DONE** - Combined residual restoration and explicit checkpoint task contracts.
- [x] **DONE** - Consistent tiled output coordinates and masked-stage unmasked-pixel preservation.
- [x] **DONE** - Guarded learned view refinement with geometric fallback.
- [x] **DONE** - Semantic defect mask contract; geometric coverage is distinct from black RGB.
- [x] **DONE** - Native paired crops, longitude wrapping and actual-join evaluation.
- [x] **DONE** - Reviewed-pair preparation and synthetic mobile capture generation.
- [x] **DONE** - Separate real-mobile metrics, identity baseline and checkpoint selection.
- [x] **DONE** - Consolidated training guide and architecture diagrams.
- [x] **DONE** - Removed obsolete steps_variable_tiled.txt. Training requirements include shared runtime requirements; export/HDR/extra metrics moved to requirements-optional.txt.
- [ ] **TODO** - Audit IMPLEMENTATION_SUMMARY.md and other historical docs before treating them as current status.
- [ ] **DEFERRED** - Partial-panorama validity-aware training and HDR/color-managed ingestion.
- [ ] **TODO** - Address the compositor's 4096-pixel working-width limit if native 12K detail is required.

The old fixed targets (500 ms classical, 200 ms GPU, under 4 GB, perfect seams)
were not established by measurement. Define thresholds on named hardware,
resolution and reviewed scenes before using them as release gates. Rotation-only
composition does not currently use translation/depth to reconstruct hidden surfaces.

## Latest checkpoint recovery implementation

- [x] **DONE** - Save restoration *_last.pt after every completed epoch with optimizer and RNG state.
- [x] **DONE** - Preserve earlier best checkpoint when resuming a worse latest epoch into a new output directory.
- [x] **DONE** - Atomically replace checkpoint files; a failed write leaves the previous checkpoint intact.
- [ ] **TODO** - Mid-epoch recovery and live AWS interruption validation remain outside this implementation.
`test_resume`, `test_mobile_adaptation` and `test_paired_panorama`: 13 tests passed on CPU, including latest recovery, preservation of the earlier best, exact resume equivalence and failed-write protection.

## Restoration fine-tuning

- [x] **DONE** - `combined.finetune` (and other restoration tasks) loads compatible weights with a fresh optimizer, epoch count and best-score selection.
- [x] **DONE** - Fine-tuning uses the new learning rate and records source checkpoint provenance in checkpoints and training reports.
- [x] **DONE** - Reject simultaneous resume/fine-tune, incompatible task/contract/channels and non-finite source weights.
- [ ] **BLOCKED** - Validate DSLR/drone-to-mobile fine-tuning quality with reviewed real mobile data; synthetic execution does not establish transfer quality.

Use a new output directory and maintain independent validation/test scenes across
both pretraining and adaptation datasets. Current split checks cannot audit
historical pretraining leakage.
Validation: 13 CPU resume/mobile/paired-data tests passed, including weight-loading, fresh optimizer/learning-rate checks and conflicting-option rejection.
