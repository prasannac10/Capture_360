# Capture360 AI Architecture

## Runtime architecture

```mermaid
flowchart TD
    A[Fisheye camera\n4-8 images] --> C[Camera metadata + poses]
    B[Phone camera\n20-30 images] --> C
    C --> D[Shared spatial ImageEncoder\nResNet18 backbone]
    D --> E[Frame mask + SetAggregator]
    D --> F[Camera-aware Spherical Projection]
    C --> F
    E --> G[Scene-conditioned spherical features]
    F --> G
    G --> H[Spatial PanoramaDecoder]
    H --> I[Initial equirectangular panorama]
    I --> J[Glare U-Net]
    J --> K[Classical lens-dot removal]
    K --> L[Nadir/Zenith mask-aware U-Net]
    L --> M[Color U-Net]
    M --> N[Classical sharpening]
    N --> O[Final panorama]
```

### Key design rules

1. **One panorama model, variable N.** The model does not assume six images. It supports 4-30 views with padding and a frame mask.
2. **Two camera geometries.** Fisheye uses 180-degree equidistant projection; phone captures use pinhole projection with calibrated intrinsics or a documented FOV-derived fallback.
3. **Shared learned representation.** The image encoder, set aggregator, spherical fusion and panorama decoder are shared across camera types.
4. **Correction models are independent.** Glare, nadir/zenith and color models have separate checkpoints and training jobs.
5. **Classical finishing stays classical.** Lens-dot removal and sharpening are deterministic post-processing, not learned models.
6. **No random correction inference.** Enabled learned correction stages require checkpoints unless explicitly overridden for experimentation.

## Training architecture

```mermaid
flowchart LR
    D[14K scene sets\nraw views + poses + camera.json + PTGUI GT] --> P[Panorama training]
    P --> PB[panorama_best.pt]
    PB --> R[Generate intermediate correction pairs]
    R --> G[glare_best.pt]
    R --> Z[nadir_zenith_best.pt]
    R --> C[color_best.pt]
    PB --> I[Integrated inference]
    G --> I
    Z --> I
    C --> I
    I --> F[Final panorama]
```

The correction models are not trained end-to-end with the panorama model in this architecture. Each stage can be evaluated and replaced independently, then composed in the inference pipeline.

## Repository boundaries

- `Pano_AI/` is the training/research and reference-inference project.
- `app/` is the Android deployment/consumer layer.
- A trained PyTorch checkpoint and exported ONNX model are artifacts, not source-controlled defaults.
