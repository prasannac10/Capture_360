# Capture360 panorama architecture

The primary workflow runs a trained **AI stitching model** first, followed by
the **correction pipeline**. AI stitching consumes calibrated source images and
produces an initial panorama; combined correction consumes that panorama as RGB
tiles. The two models require separate supervision and checkpoints. The neural
stitcher still needs held-out quality validation before production use.

See [the training guide](pano_ai/README.md) for data preparation and commands.
Diagrams below use Mermaid; open Markdown preview in a renderer with Mermaid support.
Solid arrows carry data; dotted arrows supply configuration, weights or supervision.

## 1. AI stitching first, then correction

```mermaid
flowchart TD
    frames["Native-resolution source frames"]
    tile["Overlapping source tiles<br/>Original pixel origins and dimensions"]
    encoder["Shared ResNet18 encoder<br/>Stride-8 feature maps"]
    metadata["Tile position + camera intrinsics<br/>Projection type + camera rotation"]
    tokens["Pooled visual tokens + metadata embeddings"]
    attention["Attention across tile tokens<br/>Contextual tokens + scene token"]
    second["Second streamed encoder pass<br/>Context-gated tile feature maps"]
    projection["Camera-aware spherical feature projection<br/>Weighted accumulation on feature canvas"]
    condition["Global scene conditioning"]
    decoder["Multi-scale RGB decoder<br/>Tiled output on a consistent pixel-centre grid"]
    output["Initial stitched RGB panorama<br/>Requested panorama dimensions"]
    tiles["Overlapping panorama RGB tiles"]
    correction["Correction pipeline<br/>Combined residual U-Net or separate stages"]
    merge["Blend corrected tiles"]
    final["Final corrected panorama"]
    weights["Validated correction checkpoint"]
    frames --> tile --> encoder --> tokens --> attention
    metadata --> tokens
    tile --> second
    attention -.-> second
    second --> projection --> condition --> decoder --> output
    metadata -.-> projection
    attention -.-> condition
    output --> tiles --> correction --> merge --> final
    weights -.-> correction
```

This is the default `ai_pipeline.mode: tiled_neural`, implemented by
[PanoramaModel](pano_ai/models/panorama_model.py). The stitching model has a separate training route
requiring calibrated source images and aligned reference panoramas; training
`combined` does not train this decoder. Corrected coordinate conventions require
compatible `panorama_camera_blending_v3` checkpoints.

Streaming limits feature-map memory, but attention still operates over the tile
tokens. Variable frame counts and tiled decoding do not imply unlimited memory
or validated quality at arbitrary capture counts.

The execution order is implemented in [tiled_inference.py](pano_ai/tiled_inference.py):
`PanoramaModel.forward_scene` produces `initial_panorama`, then
`HighResolutionCorrectionPipeline.run` produces the final panorama.
No classical RGB stitcher runs before the neural model in this mode.
Camera-aware projection inside the model combines features from the source views.

Train both stages with [stitching_correction_job.yaml](pano_ai/train/configs/stitching_correction_job.yaml),
which selects `tasks: [panorama, combined]`. This trains two models sequentially,
rather than a single end-to-end model. Set `inference.checkpoint` to the trained
panorama checkpoint and `correction.checkpoints.combined` to the trained
correction checkpoint. Enable `correction.toggles.combined` after validation.
With correction toggles disabled, the pipeline returns the initial neural panorama.

## 2. Data preparation, training and validation

AI stitching training is the first task in the two-stage training job:

```mermaid
flowchart TD
    sources["Calibrated source frames + camera metadata"]
    reference["Aligned reference panorama.png"]
    split["Independent physical scenes<br/>panorama/train and panorama/val"]
    model["Train PanoramaModel<br/>Source tiles to panorama RGB"]
    loss["Reference supervision<br/>L1 + SSIM + perceptual loss"]
    checkpoint["panorama_tiled_best.pt<br/>panorama_camera_blending_v3"]
    inference["Stage 1: AI stitching inference"]
    correction["Stage 2: trained correction pipeline"]
    sources --> split --> model --> loss --> checkpoint
    reference --> split
    checkpoint -.-> inference
    inference --> correction
```

The correction dataset and training are separate, as shown below. PTGui pairs
provide correction supervision; they do not supply calibrated stitching scenes.

```mermaid
flowchart TD
    pairs["Reviewed PTGui panorama<br/>+ aligned final corrected panorama"]
    manifest["Pair manifest<br/>Physical scene ID, domain, train/val split"]
    package["prepare_pairs<br/>Validate and copy paired images"]
    references["Reviewed full-sphere DSLR/drone<br/>final panoramas"]
    synth["prepare_mobile<br/>Render calibrated pinhole captures<br/>Add exposure, blur, noise and JPEG variation"]
    stitch["Geometric stitcher creates before<br/>Clean reference supplies after"]
    dataset["Prepared dataset<br/>restoration/train and restoration/val"]
    storage["Local dataset or S3 snapshot<br/>S3 downloads to local cache"]
    crops["Paired native-coordinate crops<br/>Same locations in before and after<br/>Longitude wrapping"]
    train["Train combined residual U-Net<br/>Predict corrected RGB; paired L1 loss"]
    validation["Fixed validation crops<br/>Loss, edge, texture and actual join metrics"]
    mobile["Optional reviewed real-mobile pairs<br/>Separate physical scenes in mobile_val<br/>Compare uncorrected baseline"]
    select["Select best checkpoint<br/>Mobile L1 when enabled<br/>Otherwise ordinary paired validation L1"]
    artifacts["combined_best.pt<br/>training.yaml + training_report.json"]
    review["Held-out real mobile full-panorama review<br/>Required before production activation"]
    pairs --> manifest --> package --> dataset
    references --> synth --> stitch --> dataset
    dataset --> storage --> crops --> train --> validation
    mobile --> validation
    validation --> select --> artifacts --> review
    review -.->|Approved checkpoint| inference["Inference correction block"]
```

All variants of a physical scene stay in one split. Real mobile validation scenes
must also be separate from ordinary validation. Synthetic mobile examples are
labelled `synthetic_mobile` and cannot replace a real mobile benchmark.
Synthetic generation simulates rotation and appearance changes, not translation
parallax, hidden surfaces, rolling shutter or real moving people.

The correction-only job selects `tasks: [combined]` in
[training_job.yaml](pano_ai/train/configs/training_job.yaml). This route needs
aligned before/after panoramas, not per-stage displacement, ownership or defect
mask labels. Optional alignment/blending heads require separate labelled data
and are not trained by this job. Checkpoint selection can explicitly override
the default with `combined.checkpoint_selection`.

## 3. Combined correction model: residual U-Net

```mermaid
flowchart TD
    rgb["RGB tile<br/>3 x H x W"]
    e1["Encoder 1: ConvBlock<br/>32 channels, H x W"]
    e2["MaxPool + Encoder 2<br/>64 channels, H/2 x W/2"]
    e3["MaxPool + Encoder 3<br/>128 channels, H/4 x W/4"]
    bottleneck["MaxPool + Bottleneck<br/>128 channels, H/8 x W/8"]
    d3["Upsample + concatenate Encoder 3<br/>ConvBlock: 64 channels"]
    d2["Upsample + concatenate Encoder 2<br/>ConvBlock: 32 channels"]
    d1["Upsample + concatenate Encoder 1<br/>ConvBlock: 32 channels"]
    residual["3 x 3 convolution<br/>3-channel RGB residual"]
    add["Add input RGB + residual<br/>Clamp to 0..1"]
    rgb --> e1 --> e2 --> e3 --> bottleneck --> d3 --> d2 --> d1 --> residual --> add
    e3 -->|Skip connection| d3
    e2 -->|Skip connection| d2
    e1 -->|Skip connection| d1
    rgb -->|Residual connection| add
```

Channel counts show the default `base_channels: 32`. Each ConvBlock contains two
3x3 convolutions with GroupNorm and SiLU. Upsampling is bilinear to the matching
encoder size. The implementation is
[CombinedRestorationUNet](pano_ai/models/combined_restoration.py) using
[RestorationUNet](pano_ai/models/restoration_backbone.py).
Training uses native paired crops; inference processes overlapping tiles and
blends their outputs. The default training crop is 1024x1024. No mask input is
required for this combined model, and exact preservation of unchanged regions
is not guaranteed.

## 4. Alternative: source-preserving geometric stitching

```mermaid
flowchart TD
    images["Mobile source images"]
    camera["Camera metadata<br/>capture.json or supported pose adapter<br/>Intrinsics, rotations, dimensions, filenames"]
    validate["Validate frames and camera contract"]
    project["Geometric spherical projection<br/>Rotation-only camera geometry"]
    heads["Optional trained alignment and ownership heads"]
    guard{"Proposal passes<br/>reliability checks?"}
    accept["Apply accepted view refinement"]
    fallback["Retain geometric projection<br/>and seam ownership"]
    compose["Exposure compensation<br/>Graph-cut seams and multiband blending"]
    initial["Initial panorama + geometric coverage"]
    enabled{"Validated combined<br/>correction enabled?"}
    tiles["Overlapping native RGB tiles<br/>Periodic longitude handling"]
    correction["Combined residual U-Net"]
    merge["Blend tile outputs<br/>into corrected panorama"]
    weights["combined_best.pt<br/>Checked task and checkpoint contract"]
    final["Final panorama<br/>Geometry and correction metadata"]
    images --> validate
    camera --> validate
    validate --> project
    project --> heads
    heads --> guard
    guard -->|Yes| accept
    guard -->|No or heads disabled| fallback
    accept --> compose
    fallback --> compose
    compose --> initial
    initial --> enabled
    enabled -->|Yes| tiles
    tiles --> correction --> merge --> final
    weights -.-> correction
    enabled -->|No corrections enabled| final
```

This diagram shows `ai_pipeline.mode: source_preserving` in
[config.yaml](stitching/config.yaml), implemented by
[source_inference.py](pano_ai/source_inference.py). Learned view heads are optional
and disabled by default. Their gates check confidence, displacement, consistency,
Jacobian bounds, textured overlap and photometric improvement; rejected proposals
retain the geometric result.

The calibrated classical path shares the camera contract and geometric compositor.
Mobile inference needs source frames and calibration/poses for this route, but
**does not need a corrected reference panorama**. Images alone do not supply the
camera contract automatically in this path.

Combined correction must be enabled alone among post-blend correction toggles.
The alternative separate-stage route uses explicit defect masks for masked
repairs; combined correction takes RGB only and can change any pixel. Neither
route includes an automatic person detector. Keep learned correction disabled
until real mobile evaluation supports enabling it.

## Resolution and quality limits

The requested output may be 12000x6000. In the alternative geometric route, the compositor
works at at most 4096 pixels wide and enlarges larger outputs. A 12K file therefore
does not establish native 12K detail. The AI stitching decoder uses a smaller
spherical feature canvas; output dimensions alone cannot restore lost texture.

Translations and optional depth are retained in the shared camera format, but
current composition uses rotation-only geometry. Strong nearby parallax, missing
coverage and hidden floor remain limitations. Synthetic smoke tests verify
execution and contracts; they do not demonstrate real mobile quality or parity
with a commercial panorama system.
