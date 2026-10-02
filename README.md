# Capture360

Capture360 has two independently developed components:

- **`app/`** — Android guided 360 capture application.
- **`panorama/`** — Python panorama generation, separated into `pano_ai/`, `pano_classical/`, and the configuration-driven `stitching/` dispatcher.

## Android stitching architecture

The app uses a shared capture/stitch/persist shell behind `PanoramaStitcher`. The user can switch between **OpenCV** and **AI** stitching. Captured frames retain their sensor-fusion poses; the AI path consumes those poses instead of inferring ordering from file timestamps.

Both paths pass through a shared, ordered `PanoramaCorrection` chain. The stages are independently toggleable: equirectangular finish, nadir/zenith cleanup, glare removal, dot removal, color correction, and sharpening. They are disabled/no-op by default because the available dataset contains only raw->final labels. Training correction stages requires synthetic per-step supervision and is deferred.

## Learned stitcher

`panorama/pano_ai/` expects approximately 14,000 real scene sets with `images/*.png`, `poses.pt`, and one final `panorama.png`. Run Python commands from the repository root with `python -m panorama.pano_ai.<module>`. Select `stitching.engine: ai` or `stitching.engine: classical` in `panorama/stitching/config.yaml`; callers use `panorama.stitching.stitch(...)` and are routed to the selected implementation.

The model is trained on fisheye imagery. Narrow-FOV rectilinear phone-lens adaptation is explicitly deferred and requires a new dataset with logged capture poses.

## Build

From the repository root:

```bash
./gradlew assembleDebug
```

or on Windows PowerShell:

```powershell
./gradlew.bat assembleDebug
```

The Docker build uses the root `Dockerfile`.

## AI model deployment

ONNX Runtime is the selected Android runtime. After training:

```bash
python -m panorama.pano_ai.export --config panorama/stitching/config.yaml --checkpoint panorama/pano_ai/checkpoints/simple_360.pt --output panorama/pano_ai/artifacts/pano_model.onnx
```

Copy the generated `pano_model.onnx` to `app/src/main/assets/`. `model_metadata.json` documents the fixed preprocessing and tensor contract. The repository intentionally does not contain a fake/untrained model artifact.

## Smoke test

Run from the repository root after real data is available:

```bash
python -m pytest panorama/pano_ai/tests/test_smoke.py
```

It performs a real forward/backward pass on 2–3 scene sets and asserts `[B,3,256,512]` output and finite gradients.
