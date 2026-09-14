# Capture360 Architecture

> **Repository:** `prasannac10/Capture_360`  
> **Last Updated:** 2026-09-14  
> **Status:** Phase 4 (Panorama Correction Pipeline) - COMPLETE

---

## System Overview

Capture360 is a professional 360° panorama capture, AI stitching, and correction platform with **dual-path correction** capabilities.

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPTURE360 PLATFORM                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────────┐                                        │
│  │  Android Capture │  → Images + Poses + Camera Metadata   │
│  └──────────────────┘                                        │
│            │                                                  │
│            ↓                                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │         PANO_AI CORE (Learned)                         │ │
│  │                                                         │ │
│  │  ImageEncoder → SetAggregator → SphericalFusion       │ │
│  │                        ↓                               │ │
│  │                  PanoramaDecoder                        │ │
│  │                        ↓                               │ │
│  │            Initial Panorama [256, 512, 3]             │ │
│  └────────────────────────────────────────────────────────┘ │
│            │                                                  │
│            ↓                                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  CORRECTION PIPELINE (Dual-Path: Classical + Learned) │ │
│  │                                                         │ │
│  │  Stage 1: Baseline Corrections (Learned)              │ │
│  │   • Glare Removal (GlareRemovalUNet)                  │ │
│  │   • Lens Dot Removal (Classical Morphology)           │ │
│  │   • Nadir/Zenith Inpainting (NadirZenithInpainter)   │ │
│  │   • Color Enhancement (ColorEnhancementUNet)          │ │
│  │                                                         │ │
│  │  Stage 2: Advanced Geometric (Dual-Path)              │ │
│  │   • Overlap Detection (OpenCV | OverlapDetectionUNet) │ │
│  │   • Parallax Correction (ParallaxCorrectionCV |       │ │
│  │                           ParallaxCorrectionUNet)      │ │
│  │   • Parallax Flow Detection (ParallaxCorrectionDetector)│
│  │   • Ghost Removal (GhostRemovalCV | GhostRemovalUNet) │ │
│  │                                                         │ │
│  │  Stage 3: Seam Blending (Dual-Path)                   │ │
│  │   • Seam Blending (SeamBlendingCV |                   │ │
│  │                     SeamBlendingUNet)                  │ │
│  │                                                         │ │
│  │  Stage 4: Classical Finishing                          │ │
│  │   • Sharpening (Unsharp Mask)                         │ │
│  └────────────────────────────────────────────────────────┘ │
│            │                                                  │
│            ↓                                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │           Final Panorama + Metadata                    │ │
│  │      [256, 512, 3] uint8 + Processing Info           │ │
│  └────────────────────────────────────────────────────────┘ │
│            │                                                  │
│            ↓                                                  │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Backend API → Cloud Storage → Delivery to Customer   │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Module Architecture

### Core Panorama Generation (Already Implemented)

```
Pano_AI/models/
├── encoder.py
│   └── ImageEncoder(ResNet18)
│       • Spatial feature extraction
│       • Pretrained backbone support
│       • Output: Dense feature maps [B, C, H, W]
│
├── aggregator.py
│   └── SetAggregator(dim)
│       • Permutation-invariant pooling
│       • Mask-aware weighting
│       • Output: Global context [B, dim]
│
├── spherical.py
│   └── spherical_project(features, rotations, pano_h, pano_w)
│       • Camera-aware projection
│       • Fisheye & pinhole support
│       • Visibility weighting
│       • Output: Fused features [1, C, pano_h, pano_w]
│
├── decoder.py
│   └── PanoramaDecoder(dim)
│       • Residual U-Net decoder
│       • Output: RGB panorama [1, 3, pano_h, pano_w]
│
└── panorama_model.py
    └── PanoramaModel
        • Orchestrates encoder → aggregator → fusion → decoder
        • Supports variable-N inputs
        • EMA checkpoint support
        • Output: Initial panorama + intermediate features
```

### Advanced Corrections (NEW - Phase 4)

```
Pano_AI/models/
├── advanced_corrections.py (NEW)
│   ├── ParallaxCorrectionUNet
│   │   • Learns residual parallax correction
│   │   • Inherits from RestorationUNet
│   │   • Input: Panorama [B, 3, H, W]
│   │   • Output: Corrected panorama [B, 3, H, W]
│   │
│   ├── GhostRemovalUNet
│   │   • Removes ghosting artifacts
│   │   • Supports optional mask input
│   │   • Input: Panorama [B, 3, H, W] + mask [B, 1, H, W]
│   │   • Output: Cleaned panorama [B, 3, H, W]
│   │
│   ├── SeamBlendingUNet
│   │   • Learns optimal blend weights at seams
│   │   • Input: 2 images [B, 3, H, W] + seam edges [B, 1, H, W]
│   │   • Output: Weight map [B, 1, H, W]
│   │
│   ├── OverlapDetectionUNet
│   │   • Siamese ResNet18 encoder
│   │   • 3 output heads: overlap_prob, overlap_%, confidence
│   │   • Input: 2 images [B, 3, H, W]
│   │   • Output: Scalar predictions
│   │
│   └── ParallaxCorrectionDetector
│       • Predicts 2D flow field
│       • Input: Image diff [B, 3, H, W] + mask [B, 1, H, W]
│       • Output: Flow [B, 2, H, W]
│
├── overlap_detector.py (NEW)
│   └── OverlapDetectorAI
│       • Wrapper for AI overlap detection
│       • Batch processing support
│       • Checkpoint loading
│
├── restoration_backbone.py
│   ├── ConvBlock
│   │   • Group normalization
│   │   • SiLU activation
│   │
│   └── RestorationUNet (Base class)
│       • 3-level encoder-decoder
│       • Skip connections
│       • Residual prediction mode
│       • Used by: Glare, Nadir/Zenith, Color, Parallax, Ghost
│
├── glare_removal.py (Existing)
│   └── GlareRemovalUNet
│
├── nadir_zenith.py (Existing)
│   └── NadirZenithInpainter
│
├── color_enhance.py (Existing)
│   └── ColorEnhancementUNet
│
└── lens_dots.py (Existing)
    └── remove_lens_dots()
```

### Classical Methods (NEW - Phase 4)

```
Pano_AI/stitching/
├── overlap_detector_cv.py (NEW)
│   └── OverlapDetectorCV
│       • Feature detectors: ORB, SIFT, AKAZE
│       • Lowe's ratio test matching
│       • RANSAC homography validation
│       • Methods:
│           - detect_pairwise_overlaps()
│           - detect_overlap_region()
│           - compute_overlap_statistics()
│
├── parallax_correction_cv.py (NEW)
│   └── ParallaxCorrectionCV
│       • Depth estimation: StereoSGBM
│       • Parallax interpolation: RBF/Gaussian blur
│       • Methods:
│           - estimate_parallax_shift()
│           - estimate_depth_from_stereo()
│           - correct_depth_based_parallax()
│
├── seam_blending_cv.py (NEW)
│   ├── SeamBlendingCV
│   │   • Methods: Multiband, Feather, Graph-Cut
│   │   • Seam detection: Minimum cost path DP
│   │   • Methods:
│   │       - find_seam_line()
│   │       - blend_images()
│   │       - _multiband_blend()
│   │       - _feather_blend()
│   │       - _graph_cut_blend()
│   │
│   └── GhostRemovalCV
│       • Detection: Connected components
│       • Removal: Poisson inpainting, median filter
│       • Motion analysis: Optical flow
│       • Methods:
│           - detect_ghost_regions()
│           - remove_ghosts_poisson()
│           - remove_ghosts_median_filter()
│           - detect_motion_blur()
│           - temporal_consistency_check()
│
└── opencv_stitcher.py (Existing)
    └── Classical OpenCV-based stitching
```

### Correction Pipeline (UPDATED - Phase 4)

```
Pano_AI/pipeline.py (UPDATED)
├── CorrectionPipeline
│   • Manages all correction stages
│   • Independent checkpoint loading per model
│   • 4-stage architecture:
│   │
│   │ Stage 1: Baseline Learned Corrections
│   │  ├── Glare removal (if enabled)
│   │  ├── Lens dot removal (if enabled)
│   │  ├── Nadir/zenith inpainting (if enabled)
│   │  └── Color enhancement (if enabled)
│   │
│   │ Stage 2: Advanced Geometric Corrections (Dual-Path)
│   │  ├── Overlap detection (classical or learned)
│   │  ├── Parallax correction (classical or learned)
│   │  ├── Parallax flow detection (learned only)
│   │  └── Ghost removal (classical or learned)
│   │
│   │ Stage 3: Seam-Aware Blending (Dual-Path)
│   │  └── Seam blending (classical or learned)
│   │
│   └── Stage 4: Classical Finishing
│       └── Sharpening (if enabled)
│
└── run_pipeline(images, output_path, toggles, checkpoints)
    • High-level pipeline entry point
```

### Unified Orchestration (NEW - Phase 4)

```
Pano_AI/dual_path_pipeline.py (NEW)
├── CorrectionMethodFactory
│   • create_overlap_detector()
│   • create_parallax_corrector()
│   • create_seam_blender()
│   • create_ghost_remover()
│   • Factory pattern for method selection
│
└── DualPathPanoramaPipeline
    • Orchestrates entire dual-path correction
    • Switches methods based on config
    • Methods:
        - process_panorama()
        - _detect_overlaps()
        - _correct_parallax()
        - _remove_ghosts()
        - _apply_seam_blending()
        - print_pipeline_config()
    • Output: (final_panorama, metadata)
```

### Configuration System (UPDATED)

```
Pano_AI/config.yaml (UPDATED)
├── model: Feature dimension, backbone, pano dimensions
├── input: Frame count targets, image size, normalization
├── training: Learning rate, batch size, epochs, EMA
├── loss: Geometry and supervised loss weights
│
├── correction: (Original stages - always learned)
│   ├── toggles: glare, dots, nadir_zenith, color, sharpen
│   └── checkpoints: Path to trained models
│
├── advanced_corrections: (NEW - can be classical or learned)
│   ├── toggles: parallax, ghost_removal, seam_blending,
│   │            overlap_detection, parallax_flow
│   └── checkpoints: Path to AI models
│
├── classical_methods: (NEW - enable classical methods)
│   ├── use_classical_overlap_detection
│   ├── use_classical_parallax_correction
│   ├── use_classical_seam_blending
│   └── use_classical_ghost_removal
│
├── classical_parameters: (NEW - tuning parameters)
│   ├── overlap_detection: feature_detector, min_matches
│   ├── parallax_correction: depth_method, interpolation
│   ├── seam_blending: blend_type, seam_sigma
│   └── ghost_removal: threshold_ratio, inpainting_method
│
└── inference: Output directory, format, save options
```

---

## Data Flow

### Input Processing

```
Raw Images [H, W, 3] uint8
    ↓
Normalization → [0, 1] float32
    ↓
Batching → [B, C, H, W]
    ↓
Camera Parameters Collation
    ↓
Frame Mask Generation
    ↓
Pose Matrix Creation
```

### Core Model Processing

```
[B, N, 3, 224, 224] Images
    ↓
ImageEncoder
    ↓
[B, N, 128, 28, 28] Feature Maps
    ↓
SetAggregator (Attention/pooling)
    ↓
[B, 128] Global Context
    ↓
SphericalFusion (Camera-aware projection)
    ↓
[1, 128, 256, 512] Fused Features
    ↓
PanoramaDecoder
    ↓
[1, 3, 256, 512] Initial Panorama [0, 1]
```

### Correction Pipeline Processing

```
[1, 3, 256, 512] Initial Panorama (uint8)
    ↓
┌─────────────────────────────────────────┐
│ Stage 1: Baseline Learned Corrections  │
├─────────────────────────────────────────┤
│ GlareRemovalUNet (if enabled)           │
│         ↓                               │
│ LensDots (if enabled)                   │
│         ↓                               │
│ NadirZenithInpainter (if enabled)       │
│         ↓                               │
│ ColorEnhancementUNet (if enabled)       │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│ Stage 2: Advanced Geometric Corrections (Dual-Path) │
├─────────────────────────────────────────────────────┤
│                                                     │
│ Overlap Detection:                                  │
│   IF use_classical: OverlapDetectorCV               │
│   ELSE: OverlapDetectionUNet                        │
│   Output: Overlap %, confidence, homography        │
│                                                     │
│ Parallax Correction:                                │
│   IF use_classical: ParallaxCorrectionCV            │
│   ELSE: ParallaxCorrectionUNet                      │
│   Output: Corrected panorama                        │
│                                                     │
│ Parallax Flow Detection (Learned only):             │
│   ParallaxCorrectionDetector                        │
│   Output: Flow field [2, H, W]                      │
│                                                     │
│ Ghost Removal:                                      │
│   IF use_classical: GhostRemovalCV                  │
│   ELSE: GhostRemovalUNet                            │
│   Output: Ghost-free panorama                       │
│                                                     │
└─────────────────────────────────────────────────────┘
    ↓
┌───────────────────────────────────────┐
│ Stage 3: Seam Blending (Dual-Path)   │
├───────────────────────────────────────┤
│                                       │
│ Seam Detection & Blending:            │
│   IF use_classical: SeamBlendingCV    │
│   ELSE: SeamBlendingUNet              │
│   Output: Seamlessly blended pano    │
│                                       │
└───────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 4: Classical Finishing        │
├─────────────────────────────────────┤
│ Sharpening (if enabled)             │
│         ↓                           │
│ Final Panorama (uint8)              │
└─────────────────────────────────────┘
```

---

## Model Specifications

### AI Models Summary

| Model | Location | Input | Output | Purpose |
|-------|----------|-------|--------|---------|
| **ImageEncoder** | `encoder.py` | [B,N,3,224,224] | [B,N,128,28,28] | Spatial feature extraction |
| **SetAggregator** | `aggregator.py` | [B,N,128] | [B,128] | Context aggregation |
| **PanoramaDecoder** | `decoder.py` | [1,128,256,512] | [1,3,256,512] | Panorama generation |
| **GlareRemovalUNet** | `glare_removal.py` | [1,3,256,512] | [1,3,256,512] | Glare correction |
| **NadirZenithInpainter** | `nadir_zenith.py` | [1,3,256,512] | [1,3,256,512] | Pole inpainting |
| **ColorEnhancementUNet** | `color_enhance.py` | [1,3,256,512] | [1,3,256,512] | Color/exposure correction |
| **ParallaxCorrectionUNet** | `advanced_corrections.py` | [1,3,256,512] | [1,3,256,512] | Parallax correction |
| **GhostRemovalUNet** | `advanced_corrections.py` | [1,3,256,512] | [1,3,256,512] | Ghost removal |
| **SeamBlendingUNet** | `advanced_corrections.py` | [2,3,H,W]+[1,1,H,W] | [1,1,H,W] | Blend weights |
| **OverlapDetectionUNet** | `advanced_corrections.py` | [2,3,H,W] | [1],[1],[1] | Overlap detection |
| **ParallaxCorrectionDetector** | `advanced_corrections.py` | [1,3,H,W]+[1,1,H,W] | [1,2,H,W] | Parallax flow |

### Classical Methods Summary

| Method | Location | Algorithm | Input | Output |
|--------|----------|-----------|-------|--------|
| **OverlapDetectorCV** | `overlap_detector_cv.py` | ORB/SIFT/AKAZE + RANSAC | [H,W,3] pairs | Overlap % + homography |
| **ParallaxCorrectionCV** | `parallax_correction_cv.py` | StereoSGBM + warping | [H,W,3] pairs | Corrected [H,W,3] |
| **SeamBlendingCV** | `seam_blending_cv.py` | Multiband/Feather/Graph-Cut | [H,W,3] pairs | Blended [H,W,3] |
| **GhostRemovalCV** | `seam_blending_cv.py` | Connected components + inpainting | [H,W,3] | Ghost-free [H,W,3] |

---

## Training Architecture

### Training Pipeline

```
train.py
├── Load dataset (train/val splits)
├── Build model (PanoramaModel or correction model)
├── Initialize optimizer (AdamW)
├── Setup EMA (if enabled)
│
└── For each epoch:
    ├── Training pass:
    │   ├── Forward pass
    │   ├── Compute loss (geometry or supervised)
    │   ├── Backward pass
    │   ├── Optimizer step
    │   └── EMA update
    │
    ├── Validation pass:
    │   ├── Forward pass
    │   ├── Compute loss
    │   └── Track best model
    │
    └── Save checkpoint (best + last)
```

### Loss Functions

```
Pano_AI/losses/
├── geometry.py
│   └── geometry_loss()
│       • Reprojection error
│       • Smoothness regularization
│       • Used for unsupervised training
│
└── supervised.py
    └── supervised_loss()
        • L1 loss to ground truth
        • Optional SSIM loss
        • Used when GT panorama available
```

### Checkpoint Management

```
checkpoints/
├── panorama_best.pt          [Core model]
├── panorama_last.pt          [Core model]
├── glare_best.pt             [Baseline correction]
├── nadir_zenith_best.pt      [Baseline correction]
├── color_best.pt             [Baseline correction]
│
├── parallax_best.pt          [Advanced correction - NEW]
├── ghost_removal_best.pt     [Advanced correction - NEW]
├── seam_blending_best.pt     [Advanced correction - NEW]
├── overlap_detection_best.pt [Advanced correction - NEW]
└── parallax_flow_best.pt     [Advanced correction - NEW]
```

---

## Inference Architecture

### Inference Pipeline

```
inference.py
├── Load config
├── Load dataset (validation or test)
├── Build PanoramaModel
├── Load checkpoint (with EMA support)
│
└── For each batch:
    ├── Load images, poses, camera params
    ├── Forward pass through PanoramaModel
    ├── Get initial panorama
    │
    ├── (Optional) Save intermediate features
    ├── (Optional) Save metadata (yaw/pitch)
    │
    └── Save output panorama (PNG or TIFF)
```

### Full End-to-End Inference

```
inference.py (with corrections)
├── Load PanoramaModel + checkpoint
├── Load CorrectionPipeline + all checkpoints
│
└── For each capture session:
    ├── Load images + poses + camera metadata
    ├── Forward through PanoramaModel
    │   └── Get initial panorama
    │
    ├── Forward through CorrectionPipeline
    │   ├── Stage 1: Baseline learned corrections
    │   ├── Stage 2: Advanced geometric corrections
    │   ├── Stage 3: Seam blending
    │   └── Stage 4: Classical finishing
    │   └── Get final panorama + metadata
    │
    └── Save final panorama + metadata
```

### Method Selection at Inference

```
DualPathPanoramaPipeline.process_panorama()
│
├── Read config.yaml
├── Load advanced_corrections.toggles
├── Load classical_methods flags
│
├── IF use_classical_overlap_detection:
│   └── Load OverlapDetectorCV
│ ELSE:
│   └── Load OverlapDetectionUNet
│
├── IF use_classical_parallax_correction:
│   └── Load ParallaxCorrectionCV
│ ELSE:
│   └── Load ParallaxCorrectionUNet
│
├── IF use_classical_seam_blending:
│   └── Load SeamBlendingCV
│ ELSE:
│   └── Load SeamBlendingUNet
│
├── IF use_classical_ghost_removal:
│   └── Load GhostRemovalCV
│ ELSE:
│   └── Load GhostRemovalUNet
│
└── Execute unified pipeline with selected methods
```

---

## Configuration and Control

### Method Selection Example

**Use All Classical Methods:**
```yaml
classical_methods:
  use_classical_overlap_detection: true
  use_classical_parallax_correction: true
  use_classical_seam_blending: true
  use_classical_ghost_removal: true
```

**Use All Learned Methods:**
```yaml
classical_methods:
  use_classical_overlap_detection: false
  use_classical_parallax_correction: false
  use_classical_seam_blending: false
  use_classical_ghost_removal: false

advanced_corrections:
  toggles:
    overlap_detection: true
    parallax: true
    seam_blending: true
    ghost_removal: true
    parallax_flow: true
```

**Mixed Approach:**
```yaml
classical_methods:
  use_classical_overlap_detection: true   # Fast
  use_classical_parallax_correction: false # Accurate
  use_classical_seam_blending: false      # High quality
  use_classical_ghost_removal: true       # Speed

advanced_corrections:
  toggles:
    overlap_detection: false
    parallax: true
    seam_blending: true
    ghost_removal: false
    parallax_flow: false
```

---

## Quality Metrics

### Measurement Points

```
Capture Session
    ├── Input Validation
    │   ├── Frame count: 4-8 (fisheye) or 20-30 (phone)
    │   ├── Coverage: > 360° horizontal
    │   └── Pose quality: IMU fusion validation
    │
    ├── Core Model Output
    │   ├── Initial panorama quality
    │   ├── Seam visibility
    │   └── Color consistency
    │
    ├── Correction Stage 1
    │   ├── Glare removal effectiveness
    │   ├── Nadir/zenith coverage
    │   └── Color accuracy
    │
    ├── Correction Stage 2
    │   ├── Overlap detection F1 score
    │   ├── Parallax correction RMSE
    │   ├── Ghost detection recall
    │   └── Ghost removal quality
    │
    ├── Correction Stage 3
    │   ├── Seam visibility
    │   ├── Blend smoothness
    │   └── Artifact count
    │
    └── Final Panorama
        ├── Resolution: 256×512
        ├── Format: PNG or TIFF
        └── Metadata: Yaw/pitch angles + camera params
```

---

## Performance Targets

| Component | Operation | Target | Status |
|-----------|-----------|--------|--------|
| **Core Model** | Training | < 10 hours (8 epochs) | Pending |
| **Correction Models** | Training | < 2 hours each | Pending |
| **CPU Inference** | Classical methods | < 500ms | Pending benchmark |
| **GPU Inference** | Learned models | < 200ms | Pending benchmark |
| **Memory Usage** | GPU inference | < 4GB | Pending benchmark |
| **Overlap Detection F1** | Accuracy | > 0.9 | Pending training |
| **Parallax Correction** | RMSE | < 5 pixels | Pending training |
| **Ghost Removal** | SSIM | > 0.95 | Pending training |

---

## File Organization

```
Capture_360/
├── Pano_AI/
│   ├── models/
│   │   ├── __init__.py
│   │   ├── encoder.py                    [Core]
│   │   ├── aggregator.py                 [Core]
│   │   ├── decoder.py                    [Core]
│   │   ├── spherical.py                  [Core]
│   │   ├── panorama_model.py             [Core]
│   │   ├── restoration_backbone.py       [Base for corrections]
│   │   ├── glare_removal.py              [Baseline correction]
│   │   ├── nadir_zenith.py               [Baseline correction]
│   │   ├── color_enhance.py              [Baseline correction]
│   │   ├── lens_dots.py                  [Baseline correction]
│   │   ├── advanced_corrections.py       [NEW - 5 AI models]
│   │   └── overlap_detector.py           [NEW - AI overlap]
│   │
│   ├── stitching/
│   │   ├── __init__.py
│   │   ├── opencv_stitcher.py            [Baseline classical]
│   │   ├── overlap_detector_cv.py        [NEW - Classical overlap]
│   │   ├── parallax_correction_cv.py     [NEW - Classical parallax]
│   │   └── seam_blending_cv.py           [NEW - Classical seam + ghost]
│   │
│   ├── data/
│   │   ├── dataset.py
│   │   ├── collate.py
│   │   └── transform.py
│   │
│   ├── losses/
│   │   ├── geometry.py
│   │   └── supervised.py
│   │
│   ├── utils/
│   │   ├── checkpoint.py
│   │   ├── ema.py
│   │   └── pose.py
│   │
│   ├── sharpen.py                        [Finishing]
│   ├── pipeline.py                       [UPDATED - 4-stage]
│   ├── dual_path_pipeline.py             [NEW - Orchestration]
│   ├── inference.py                      [Inference entry]
│   ├── train.py                          [Training entry]
│   ├── config.yaml                       [UPDATED - Config]
│   └── requirements.txt                  [Dependencies]
│
├── IMPLEMENTATION_SUMMARY.md             [NEW - Full docs]
├── architecture.md                       [This file - UPDATED]
├── plan.md                               [UPDATED - Status]
└── README.md
```

---

## Technology Stack

### Core Framework
- **PyTorch 2.0+** - Deep learning
- **NumPy** - Numerical computing
- **OpenCV** - Classical vision algorithms
- **PIL** - Image I/O
- **YAML** - Configuration
- **SciPy** - Scientific computing

### Android (Capture)
- **Kotlin** - Primary language
- **Camera2 API** - Camera access
- **Sensor Framework** - IMU/orientation
- **Jetpack Compose** - UI framework

### Backend (Phase 6+)
- **Python FastAPI** or **Django** - API framework
- **PostgreSQL** - Relational data
- **Redis** - Caching/queues
- **Cloud Storage** - S3-compatible (AWS/Google Cloud/Azure)
- **Container** - Docker

---

## Deployment Targets

### Phase 5: Python Reference
- **Environment:** Linux/macOS/Windows
- **GPU Support:** CUDA 11.8+, cuDNN 8.x
- **Python:** 3.9+
- **Memory:** 8GB+ (CPU inference), 4GB+ (GPU inference)

### Phase 6: Cloud Inference
- **Container:** Docker/ONNX
- **Hardware:** GPU (A100, V100, T4)
- **Framework:** TensorRT, ONNX Runtime
- **Latency Target:** < 500ms/panorama

### Phase 10: Android Capture App
- **Min SDK:** 24
- **Target SDK:** 34
- **Device:** Modern smartphones (2020+)
- **RAM:** 4GB+

---

## Summary

**Capture360** is now a **dual-path panorama processing platform** with:

✅ **Core Model:** Learned panorama generation (ImageEncoder → Decoder)  
✅ **Baseline Corrections:** 4 learned stages (glare, nadir, color, polish)  
✅ **Advanced Corrections:** 5 new learnable stages (parallax, ghost, seam, overlap, flow)  
✅ **Classical Methods:** 4 OpenCV implementations for comparison  
✅ **Unified Pipeline:** Factory pattern enables method switching  
✅ **Configuration:** YAML-driven stage toggles and parameters  

**Next Phase:** Train the 5 new AI models (M3) and benchmark against classical methods.
