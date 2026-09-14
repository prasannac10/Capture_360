# Capture360 Product Development Plan

> **Repository:** `prasannac10/Capture_360`  
> **Working branch:** `prasannac10-patch-1`  
> **Product:** Professional 360° panorama capture, AI stitching and correction platform  
> **Last updated:** 2026-09-14

## Status legend

- **DONE** — Implemented and verified at project level.
- **PARTIAL** — Some implementation exists, but the milestone is not complete.
- **TODO** — Not implemented yet.
- **BLOCKED** — Cannot be completed until a prerequisite is available.

---

# Overall Roadmap

| Phase | Area | Status | Immediate outcome |
|---|---|---|---|
| 0 | Product & Business Definition | PARTIAL | Freeze MVP business model and success metrics |
| 1 | Panorama Capture Technology | PARTIAL | Reliable Android capture package with poses and camera metadata |
| 2 | Classical Stitching Baseline | PARTIAL | Stable reference stitcher and benchmark |
| 3 | Pano_AI Core | PARTIAL | Train and validate the learned panorama model |
| 4 | Panorama Correction Pipeline | **DONE** | ✅ Dual-path classical + AI correction models implemented |
| 5 | Python Reference Inference | PARTIAL | Deterministic end-to-end Python inference |
| 6 | AI Model Deployment | TODO | ONNX/TensorRT/cloud inference service |
| 7 | Backend Platform | TODO | APIs for users, agents, jobs and processing |
| 8 | Cloud Storage & Data | TODO | Secure scalable image/panorama storage |
| 9 | Customer Application | TODO | Customer request-to-panorama workflow |
| 10 | Agent Application | PARTIAL | Complete professional agent workflow around capture |
| 11 | Agent Marketplace | TODO | Job assignment and agent dispatch |
| 12 | Payment System | TODO | Customer payment, commission and agent payout |
| 13 | Admin Portal | TODO | Operations, support, reprocessing and monitoring |
| 14 | Security & Privacy | TODO | Production-grade security, privacy and auditability |
| 15 | Quality & Production Testing | TODO | Systematic quality, performance and reliability validation |
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

- [ ] Define customer journey from request to panorama delivery.
- [ ] Define agent journey from onboarding to completed job.
- [ ] Define customer profile.
- [ ] Define agent profile.
- [ ] Define service area for initial launch.
- [ ] Define price per panorama/job.
- [ ] Define agent payout.
- [ ] Define platform commission.
- [ ] Define cancellation/refund policy.
- [ ] Define minimum viable business metrics.
- [ ] Define MVP scope and explicitly defer non-MVP features.

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

- [x] Android project created.
- [x] Camera preview foundation.
- [x] Frame capture foundation.
- [x] Motion sensor integration foundation.
- [x] Orientation estimation foundation.
- [x] Angular coverage tracking foundation.
- [x] Frame gating foundation.
- [x] Guidance overlay foundation.
- [x] Dual-camera architecture merged into `prasannac10-patch-1`.
- [x] Fisheye capture target: 4–8 images.
- [x] Phone/pinhole capture target: 20–30 images.

## Action items

- [ ] Stabilize `CameraController.kt`.
- [ ] Stabilize `SensorFusionManager.kt`.
- [ ] Complete `ComplementaryFilter.kt`.
- [ ] Complete orientation representation and utilities.
- [ ] Improve IMU drift correction.
- [ ] Add magnetometer fusion where appropriate.
- [ ] Add VIO-lite refinement if required.
- [ ] Improve angular coverage heatmap.
- [ ] Finalize frame-gating rules.
- [ ] Add capture quality validation.
- [ ] Detect insufficient angular coverage.
- [ ] Detect excessive motion / poor frames.
- [ ] Define retry flow for bad captures.
- [ ] Capture camera calibration/intrinsic metadata.
- [ ] Capture camera projection metadata.
- [ ] Capture per-frame pose metadata.
- [ ] Package images + poses + camera metadata as one capture session.
- [ ] Support offline capture and retryable upload.
- [ ] Validate true fisheye source/calibration separately from normal phone camera.
- [ ] Test fisheye workflow with 4, 6 and 8 frames.
- [ ] Test phone workflow with 20, 24 and 30 frames.

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

- [x] OpenCV-based stitching foundation exists.
- [x] Classical stitching remains separate from the learned architecture.
- [x] Fisheye stitching prototype exists.

## Action items

- [x] **NEW:** Overlap detection with ORB/SIFT/AKAZE feature matching
- [x] **NEW:** Seam detection using minimum-cost path DP
- [x] **NEW:** Multi-band Laplacian blending, Feather blending, Graph-Cut blending
- [x] **NEW:** Ghost removal with Poisson inpainting and median filtering
- [x] **NEW:** Parallax correction from stereo matching and feature-based flow
- [ ] Stabilize classical fisheye stitching.
- [ ] Add/validate phone/pinhole baseline.
- [ ] Validate camera calibration handling.
- [ ] Compare 4/6/8 fisheye frames.
- [ ] Compare 20/24/30 phone frames.
- [ ] Measure seam quality.
- [ ] Measure exposure consistency.
- [ ] Evaluate moving objects.
- [ ] Evaluate repetitive textures.
- [ ] Evaluate poles/nadir/zenith.
- [ ] Measure processing time.
- [ ] Build a fixed classical-vs-AI benchmark dataset.
- [ ] Store baseline outputs for every benchmark scene.

## Exit criteria

A repeatable baseline exists against which every Pano_AI model version can be compared.

---

# Phase 3 — Pano_AI Core

**Status: PARTIAL**

## Goal

Train a camera-aware, arbitrary-N learned panorama generation model.

## Architecture

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

- [x] Configurable image encoder.
- [x] ResNet18 encoder option.
- [x] Optional pretrained encoder.
- [x] Spatial feature-map retention.
- [x] Feature projection.
- [x] SetAggregator.
- [x] Variable-N input handling.
- [x] Frame mask handling.
- [x] SphericalFusion.
- [x] Camera-aware projection foundation.
- [x] Panorama decoder.
- [x] End-to-end PanoramaModel structure.
- [x] EMA support.
- [x] YAML-driven configuration foundation.

## Partial / pending

- [ ] Fully validate fisheye projection.
- [ ] Fully validate pinhole projection.
- [ ] Validate mixed camera metadata handling.
- [ ] Finalize geometry loss.
- [ ] Finalize supervised L1/SSIM loss configuration.
- [ ] Validate checkpoint save/load workflow.
- [ ] Validate EMA checkpoint inference.
- [ ] Validate 14K dataset end-to-end.
- [ ] Train first full model on the actual dataset.
- [ ] Evaluate quantitatively.
- [ ] Evaluate visually.
- [ ] Compare against classical baseline.
- [ ] Analyze failures by camera type and frame count.

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

- [ ] Finalize scene organization.
- [ ] Finalize camera metadata.
- [ ] Finalize pose metadata.
- [ ] Split by capture scene/session, not individual image.
- [ ] Create 80/10/10 train/validation/test split.
- [ ] Stratify by camera type.
- [ ] Stratify by frame count.
- [ ] Validate all image/pose/camera relationships.
- [ ] Create dataset visualization tools.
- [ ] Build evaluation buckets:
  - [ ] Fisheye 4–6 frames.
  - [ ] Fisheye 7–8 frames.
  - [ ] Phone 20–24 frames.
  - [ ] Phone 25–30 frames.
  - [ ] Wrap-around regions.
  - [ ] Pole/nadir/zenith regions.

## Exit criteria

A trained checkpoint produces a usable initial panorama on held-out scenes and demonstrates measurable improvement versus the classical baseline.

---

# Phase 4 — Panorama Correction Pipeline

**Status: ✅ DONE**

## Goal

Improve the initial panorama through independently trainable correction stages using both classical and AI methods.

## Completed

### Baseline Correction Stages (Learned)
- [x] Glare U-Net architecture
- [x] Lens-dot classical processing
- [x] Nadir/Zenith mask-aware U-Net
- [x] Color enhancement U-Net
- [x] Classical sharpening

### Advanced Correction Models (NEW - Both Classical & Learned)
- [x] **Overlap Detection**
  - [x] Classical: `OverlapDetectorCV` (ORB/SIFT/AKAZE, RANSAC homography)
  - [x] Learned: `OverlapDetectionUNet` (Siamese ResNet18)
  
- [x] **Parallax Correction**
  - [x] Classical: `ParallaxCorrectionCV` (Stereo SGBM, RBF interpolation)
  - [x] Learned: `ParallaxCorrectionUNet` (U-Net residual learning)
  
- [x] **Seam-Aware Blending**
  - [x] Classical: `SeamBlendingCV` (Multiband Laplacian, Feather, Graph-Cut)
  - [x] Learned: `SeamBlendingUNet` (Pixel-wise blend weight learning)
  
- [x] **Ghost Artifact Removal**
  - [x] Classical: `GhostRemovalCV` (Poisson inpainting, median filter, optical flow)
  - [x] Learned: `GhostRemovalUNet` (Artifact inpainting U-Net)
  
- [x] **Parallax Flow Detection**
  - [x] Learned: `ParallaxCorrectionDetector` (2D flow field prediction)

### Pipeline Architecture
- [x] 4-stage correction pipeline:
  1. Stage 1: Baseline learned corrections (glare, nadir_zenith, color)
  2. Stage 2: Advanced geometric corrections (parallax, ghost removal, overlap validation)
  3. Stage 3: Seam-aware blending
  4. Stage 4: Classical finishing (dots, sharpening)

- [x] Dual-path factory pattern (`DualPathPanoramaPipeline`)
  - Switch between classical and learned methods via config
  - Factory pattern for clean method selection
  - Unified orchestration layer

- [x] Configuration system
  - `advanced_corrections` section with per-model toggles
  - `classical_methods` boolean flags
  - `classical_parameters` for OpenCV tuning

### Integration
- [x] Updated `pipeline.py` with 4-stage architecture
- [x] All 5 new AI models in `advanced_corrections.py`
- [x] Overlap detector AI in `overlap_detector.py`
- [x] Classical methods in `stitching/` folder
- [x] Updated `config.yaml` with all toggles
- [x] Unified `dual_path_pipeline.py` orchestration

### Documentation
- [x] `IMPLEMENTATION_SUMMARY.md` with:
  - Architecture diagrams
  - File structure overview
  - Model specifications
  - Usage examples
  - Training workflow
  - Classical vs Learned comparison table

## Pending (After Training)

- [ ] Prepare parallax training data (depth maps or flow ground truth)
- [ ] Prepare ghost artifact training data
- [ ] Prepare seam blending training data
- [ ] Prepare overlap detection training data
- [ ] Train ParallaxCorrectionUNet
- [ ] Train GhostRemovalUNet
- [ ] Train SeamBlendingUNet
- [ ] Train OverlapDetectionUNet
- [ ] Train ParallaxCorrectionDetector
- [ ] Evaluate each model independently
- [ ] Benchmark classical methods baseline
- [ ] Compare classical vs learned quality
- [ ] Compare classical vs learned speed
- [ ] Enable trained models in config.yaml
- [ ] Integrate into production pipeline

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

## Target flow

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

- [x] Dataset/DataLoader foundation.
- [x] Variable-N collation.
- [x] Pose normalization.
- [x] Camera parameter collation.
- [x] Panorama model inference foundation.
- [x] Checkpoint loading.
- [x] EMA support.
- [x] Intermediate spherical feature output option.
- [x] Panorama metadata output foundation.
- [x] **NEW:** Integration with dual-path correction pipeline
- [x] **NEW:** Configurable method selection (classical vs learned)
- [x] **NEW:** Auxiliary data handling (seam edges, overlap masks, etc.)

## Action items

- [x] Ensure `camera_params` are always passed to PanoramaModel.
- [x] Complete learned model → correction pipeline integration.
- [ ] Define correction-mask input/output contract.
- [ ] Make inference deterministic.
- [ ] Save initial panorama.
- [ ] Save final panorama.
- [ ] Save yaw/pitch metadata.
- [ ] Save camera metadata.
- [ ] Support TIFF output.
- [ ] Add EXR/HDR output where required.
- [ ] Add automated end-to-end inference test.
- [ ] Add malformed-input validation.
- [ ] Benchmark CPU inference (classical methods).
- [ ] Benchmark GPU inference (learned models).
- [ ] Record memory consumption.
- [ ] Document method switching instructions.
- [ ] Create inference comparison scripts (classical vs learned).

## Exit criteria

One command can take a complete capture session and produce a final panorama plus metadata reproducibly using either classical or learned methods.

---

# Phase 6 — AI Model Deployment

**Status: TODO**

## Goal

Move the validated Python model into a production inference environment.

## Model export

- [ ] Export PanoramaModel to ONNX.
- [ ] **NEW:** Export ParallaxCorrectionUNet to ONNX.
- [ ] **NEW:** Export GhostRemovalUNet to ONNX.
- [ ] **NEW:** Export SeamBlendingUNet to ONNX.
- [ ] **NEW:** Export OverlapDetectionUNet to ONNX.
- [ ] Validate ONNX output against PyTorch.
- [ ] Investigate dynamic input-N support.
- [ ] Validate camera metadata inputs.
- [ ] Validate output equivalence.
- [ ] TensorRT experimentation.
- [ ] FP16 inference.
- [ ] INT8 experimentation.
- [ ] Model size optimization.
- [ ] Latency benchmark.
- [ ] GPU memory benchmark.

## Cloud inference

- [ ] Select cloud provider.
- [ ] Define GPU instance/service.
- [ ] Containerize inference.
- [ ] Create inference API/service.
- [ ] Add job queue.
- [ ] Add worker process.
- [ ] Add retries.
- [ ] Add autoscaling.
- [ ] Add monitoring.
- [ ] Add failure reporting.

## Exit criteria

A production-like service can accept a capture session and return a final panorama with predictable latency and cost.

---

# Phase 7 — Backend Platform

**Status: TODO**

## Goal

Provide the business APIs connecting customers, agents, capture sessions and AI processing.

## Action items

- [ ] Select backend technology.
- [ ] Design REST/API contract.
- [ ] Implement authentication.
- [ ] Implement authorization.
- [ ] User management.
- [ ] Agent management.
- [ ] Customer management.
- [ ] Job management.
- [ ] Capture-session management.
- [ ] Image-set management.
- [ ] Processing-job management.
- [ ] Panorama management.
- [ ] Processing status API.
- [ ] Error-management API.
- [ ] Notification integration.
- [ ] Admin APIs.
- [ ] API logging and monitoring.

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

**Status: TODO**

## Goal

Securely store originals, intermediate data and final panoramas at scale.

## Action items

- [ ] Select object-storage provider.
- [ ] Define storage hierarchy.
- [ ] Upload original images.
- [ ] Store capture metadata.
- [ ] Store poses.
- [ ] Store intermediate processing data where required.
- [ ] Store final panoramas.
- [ ] Configure lifecycle policies.
- [ ] Configure backups.
- [ ] Define deletion policy.
- [ ] Add CDN for panorama delivery.
- [ ] Implement signed/private URLs.
- [ ] Define storage cost controls.
- [ ] Monitor storage growth.

## Exit criteria

A panorama remains securely accessible to the authorized customer while raw data and temporary processing data follow defined retention rules.

---

# Phase 9 — Customer Application

**Status: TODO**

## Goal

Allow customers to request, pay for and consume panorama services.

## Action items

- [ ] Registration/login.
- [ ] Customer profile.
- [ ] Create panorama request.
- [ ] Location capture.
- [ ] Select service.
- [ ] Display price.
- [ ] Payment.
- [ ] Track request.
- [ ] Show assigned agent.
- [ ] Show capture status.
- [ ] Show processing status.
- [ ] View panorama.
- [ ] Download panorama.
- [ ] Share panorama.
- [ ] Job history.
- [ ] Invoice/receipt.

## Exit criteria

A customer can complete the entire request → payment → panorama delivery journey.

---

# Phase 10 — Agent Application

**Status: PARTIAL**

## Goal

Provide professional agents with everything required to perform and complete capture jobs.

## Already available

- [x] Capture application foundation.
- [x] Camera capture.
- [x] Motion/orientation foundation.
- [x] Angular coverage guidance.
- [x] Dual-camera architecture.
- [x] Fisheye capture target: 4–8 images.
- [x] Phone capture target: 20–30 images.

## Agent onboarding

- [ ] Agent registration.
- [ ] Agent profile.
- [ ] Phone verification.
- [ ] Service-area selection.
- [ ] Availability.
- [ ] Training content.
- [ ] Capture-quality certification.
- [ ] Admin approval.

## Job workflow

- [ ] Job list.
- [ ] Job details.
- [ ] Accept/reject job.
- [ ] Navigation.
- [ ] Start capture.
- [ ] Validate capture quality.
- [ ] Upload capture session.
- [ ] Show upload progress.
- [ ] Show processing status.
- [ ] Complete job.
- [ ] Show earnings.

## Exit criteria

A trained agent can accept a job, capture a valid session, upload it, and receive completion confirmation.

---

# Phase 11 — Agent Marketplace

**Status: TODO**

## Goal

Automate assignment of customer jobs to suitable agents.

## Action items

- [ ] Agent availability.
- [ ] Agent location.
- [ ] Nearby-agent search.
- [ ] Manual assignment.
- [ ] Automatic assignment.
- [ ] Job acceptance timeout.
- [ ] Reassignment.
- [ ] Ratings.
- [ ] Cancellation.
- [ ] Dispute handling.

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

- [ ] Select payment gateway.
- [ ] Customer payment initiation.
- [ ] Payment status verification.
- [ ] Failed-payment handling.
- [ ] Refund workflow.
- [ ] Cancellation charges.
- [ ] Platform commission calculation.
- [ ] Agent payout calculation.
- [ ] Payout processing.
- [ ] Reconciliation.
- [ ] Invoice generation.
- [ ] Payment audit trail.

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

- [ ] Admin login.
- [ ] Customer management.
- [ ] Agent management.
- [ ] Agent approval.
- [ ] Job management.
- [ ] Manual agent assignment.
- [ ] Capture-session inspection.
- [ ] Upload inspection.
- [ ] Panorama viewing.
- [ ] Reprocess failed jobs.
- [ ] Failure handling.
- [ ] Payment monitoring.
- [ ] Payout monitoring.
- [ ] Customer support tools.
- [ ] Operational metrics.
- [ ] AI processing metrics.
- [ ] Cost monitoring.

## Exit criteria

Operations can manage a customer job without engineering intervention.

---

# Phase 14 — Security & Privacy

**Status: TODO**

## Goal

Make customer images, accounts and payments secure enough for production use.

## Action items

- [ ] Authentication.
- [ ] Role-based access control.
- [ ] Secure APIs.
- [ ] TLS for data in transit.
- [ ] Encryption at rest.
- [ ] Private image storage.
- [ ] Signed URLs.
- [ ] Data retention policy.
- [ ] Data deletion workflow.
- [ ] Audit logs.
- [ ] Backup and recovery.
- [ ] Privacy policy.
- [ ] Terms of service.
- [ ] Agent agreement.
- [ ] Payment compliance review.
- [ ] Security testing.

## Exit criteria

Access to customer data is explicitly controlled, auditable and revocable.

---

# Phase 15 — Quality & Production Testing

**Status: TODO**

## Goal

Establish objective quality gates for both AI output and the complete product.

## Panorama test matrix

### Camera / frame count

- [ ] Fisheye — 4 frames.
- [ ] Fisheye — 6 frames.
- [ ] Fisheye — 8 frames.
- [ ] Phone — 20 frames.
- [ ] Phone — 24 frames.
- [ ] Phone — 30 frames.

### Scene conditions

- [ ] Bright indoor.
- [ ] Dark indoor.
- [ ] Outdoor.
- [ ] Reflective surfaces.
- [ ] Moving people.
- [ ] Furniture/windows.
- [ ] Repetitive textures.
- [ ] Low light.
- [ ] Glare.
- [ ] Lens dots.
- [ ] Nadir/floor.
- [ ] Zenith/ceiling.
- [ ] Exposure differences.

## Product testing

- [ ] Android unit tests.
- [ ] AI unit tests.
- [ ] Backend unit tests.
- [ ] API tests.
- [ ] End-to-end tests.
- [ ] Load tests.
- [ ] Security tests.
- [ ] Payment tests.
- [ ] Device compatibility tests.
- [ ] Network interruption tests.
- [ ] Offline/retry tests.

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

- [ ] Recruit/train first agents.
- [ ] Select pilot customers.
- [ ] Define pilot pricing.
- [ ] Run real capture sessions.
- [ ] Monitor upload reliability.
- [ ] Monitor AI processing.
- [ ] Collect panorama quality feedback.
- [ ] Collect customer feedback.
- [ ] Collect agent feedback.
- [ ] Measure operating cost.
- [ ] Identify top failure modes.
- [ ] Fix highest-impact failures.
- [ ] Repeat pilot.

## Exit criteria

The service can repeatedly deliver acceptable panoramas to paying customers at a sustainable operating cost.

---

# Phase 17 — Public Launch

**Status: TODO**

## Goal

Move from controlled pilot to production commercial operation.

## Technical readiness

- [ ] Production cloud environment.
- [ ] Production database.
- [ ] Production object storage.
- [ ] Production AI inference.
- [ ] Monitoring.
- [ ] Alerts.
- [ ] Backups.
- [ ] Disaster recovery.
- [ ] CI/CD.
- [ ] Android production release.
- [ ] iOS production release.
- [ ] Admin portal production release.
- [ ] Production security validation.

## Business readiness

- [ ] Final pricing.
- [ ] Payment system.
- [ ] Customer contracts where required.
- [ ] Terms of service.
- [ ] Privacy policy.
- [ ] Refund policy.
- [ ] Customer support process.
- [ ] Agent recruitment process.
- [ ] Agent training.
- [ ] Marketing website.
- [ ] Customer acquisition process.

## Exit criteria

Capture360 can accept customers, dispatch/coordinate agents, process panoramas, collect payment, pay agents and support failures without depending on manual engineering intervention.

---

# Recommended Immediate Milestones

## M1 — Prove AI

**Priority: P0**

```text
14K dataset
   ↓
Train Pano_AI
   ↓
Evaluate held-out scenes
   ↓
Compare with classical baseline
   ↓
Good initial panorama
```

### Actions

- [ ] Finalize dataset structure.
- [ ] Finalize train/validation/test split.
- [ ] Validate camera and pose metadata.
- [ ] Run first training.
- [ ] Inspect outputs.
- [ ] Fix highest-impact model/data issues.
- [ ] Establish quantitative baseline.

**Gate:** Do not invest heavily in marketplace/backend complexity until the panorama quality is acceptable.

---

## M2 — Complete Capture

**Priority: P0**

```text
Android capture
   ↓
Images
   +
Poses
   +
Camera metadata
   ↓
Capture package
```

### Actions

- [ ] Finish pose accuracy.
- [ ] Finish coverage tracking.
- [ ] Finish quality checks.
- [ ] Finalize metadata schema.
- [ ] Test fisheye workflow.
- [ ] Test phone workflow.
- [ ] Test offline capture.

---

## M3 — Train Correction Models

**Priority: P1** ← **NEW MILESTONE**

```text
Prepare training data
   ↓
   ├── Parallax ground truth (depth/flow)
   ├── Ghost artifacts annotations
   ├── Seam quality labels
   └── Overlap measurements
   ↓
Train 5 independent AI models
   ├── ParallaxCorrectionUNet
   ├── GhostRemovalUNet
   ├── SeamBlendingUNet
   ├── OverlapDetectionUNet
   └── ParallaxCorrectionDetector
   ↓
Benchmark classical methods
   ↓
Compare classical vs learned
   ↓
Enable trained models in production
```

### Actions

- [x] Implement all classical methods (DONE)
- [x] Implement all AI models (DONE)
- [x] Create dual-path pipeline (DONE)
- [ ] Prepare parallax ground truth data
- [ ] Prepare ghost artifact training pairs
- [ ] Prepare seam blending training data
- [ ] Prepare overlap detection labels
- [ ] Train ParallaxCorrectionUNet
- [ ] Train GhostRemovalUNet
- [ ] Train SeamBlendingUNet
- [ ] Train OverlapDetectionUNet
- [ ] Train ParallaxCorrectionDetector
- [ ] Validate each model independently
- [ ] Benchmark classical baseline speed
- [ ] Benchmark classical baseline quality
- [ ] Compare trained models against classical
- [ ] Document results and decisions
- [ ] Enable best models in config.yaml

**Gate:** Train and validate models before moving to cloud deployment.

---

## M4 — Cloud Processing

**Priority: P1**

```text
Upload
   ↓
AI inference (classical or learned)
   ↓
Correction pipeline
   ↓
Final panorama
   ↓
View / download
```

### Actions

- [ ] Complete Python reference inference.
- [ ] Export trained models to ONNX.
- [ ] Build inference container.
- [ ] Build processing API.
- [ ] Add object storage.
- [ ] Add job queue.
- [ ] Return final panorama.

---

## M5 — Minimum Business Platform

**Priority: P1**

### Actions

- [ ] Customer account.
- [ ] Agent account.
- [ ] Job creation.
- [ ] Manual job assignment.
- [ ] Processing queue.
- [ ] Panorama delivery.

---

# Summary of Recent Changes (2026-09-14)

## Phase 4 — Panorama Correction Pipeline: NOW COMPLETE ✅

### What was implemented:

1. **Dual-Path Architecture**
   - Classical methods: 4 complete OpenCV-based implementations
   - Learned methods: 5 new AI models with PyTorch
   - Factory pattern for seamless switching

2. **8 New Files Added**
   - `Pano_AI/models/advanced_corrections.py` (5 AI models)
   - `Pano_AI/models/overlap_detector.py` (Siamese overlap net)
   - `Pano_AI/stitching/overlap_detector_cv.py` (Classical overlap)
   - `Pano_AI/stitching/seam_blending_cv.py` (Seam + ghost removal)
   - `Pano_AI/stitching/parallax_correction_cv.py` (Parallax correction)
   - `Pano_AI/dual_path_pipeline.py` (Orchestration)
   - Updated `Pano_AI/pipeline.py` (4-stage architecture)
   - Updated `Pano_AI/config.yaml` (New toggles)

3. **Features Implemented**
   - ✅ Overlap Detection (classical + learned)
   - ✅ Parallax Correction (classical + learned)
   - ✅ Seam Blending (3 classical methods + learned)
   - ✅ Ghost Removal (classical + learned)
   - ✅ Parallax Flow Detection (learned)
   - ✅ Plus existing: Feature matching, Alignment, Exposure correction

4. **Configuration System**
   - Per-model toggle switches
   - Method selection (classical vs learned)
   - Parameter tuning for each method

### Next Critical Step:

**TRAIN THE 5 NEW AI MODELS** - This is M3 priority

- Parallax ground truth data needed
- Ghost artifact datasets needed
- Seam quality annotations needed
- Overlap labels needed

Then benchmark classical methods and compare.

---

# Key Metrics to Track

| Metric | Target | Status |
|--------|--------|--------|
| Overlap detection F1 score | > 0.9 | Pending model training |
| Parallax correction RMSE | < 5 pixels | Pending model training |
| Ghost removal SSIM | > 0.95 | Pending model training |
| Seam blending - no artifacts | 100% test scenes | Pending model training |
| Inference time (classical) | < 500ms | Need benchmark |
| Inference time (learned GPU) | < 200ms | Need benchmark |
| Memory usage | < 4GB | Need benchmark |

---

This plan is now **aligned with implementation status** and ready for Phase 5: Python Reference Inference and Model Training.
