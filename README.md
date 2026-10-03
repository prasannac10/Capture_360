# Capture360

Capture360 has two independently developed components:

- **`app/`** — Android guided 360 capture application.
- **`panorama/`** — Python panorama generation, separated into `pano_ai/`, `pano_classical/`, and the configuration-driven `stitching/` dispatcher.

## Android stitching architecture

The app uses a shared capture/stitch/persist shell behind `PanoramaStitcher`. The user can switch between **OpenCV** and **AI** stitching. Captured frames retain their sensor-fusion poses; the AI path consumes those poses instead of inferring ordering from file timestamps.

Both paths pass through a shared, ordered `PanoramaCorrection` chain. The stages are independently toggleable: equirectangular finish, nadir/zenith cleanup, glare removal, dot removal, color correction, and sharpening. They are disabled/no-op by default because the available dataset contains only raw->final labels. Training correction stages requires synthetic per-step supervision and is deferred.

## Learned stitcher

The Python default is `tiled_neural` with native RGB residual detail. It consumes
calibrated fisheye or pinhole source frames and produces 12000x6000 RGB before
optional correction. Training needs aligned, genuine 12K reference panoramas;
no fixed scene count establishes quality. Run Python commands from the repository
root with `python -m panorama.pano_ai.<module>`. Select the engine in
`panorama/stitching/config.yaml`; callers use `panorama.stitching.stitch(...)`.

Drone/mobile profiles accept originals up to 3500x2100 in either orientation,
alongside their original supported sizes. Accurate per-frame calibration and
poses are required. The new detail head requires retraining with
`panorama_native_rgb_residual_v6` checkpoints; existing v4 weights cannot supply
it. See the [detailed architecture](panorama/ARCHITECTURE_VARIABLE_TILED.md),
[training guide](panorama/pano_ai/README.md), and
[testing guide](panorama/stitching/TESTING.md#neural-native-detail-validation).

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

The native tiled Python model is the reference implementation. Its tile stream,
spherical projection and native RGB sampler are not currently exported as an
Android ONNX graph. The export command writes runtime-contract notes only:

```bash
python -m panorama.pano_ai.export --output panorama/pano_ai/artifacts/tiled_runtime_contract.json
```

This JSON is not a trained model artifact. Android integration needs a compatible
deployment implementation; the repository does not supply a fake/untrained ONNX model.

## Smoke test

Run synthetic AI regression checks from the repository root:

```bash
python -B -m unittest panorama.pano_ai.tests.test_native_detail
python -B -m unittest discover -s panorama/pano_ai/tests
```

These verify model/data contracts, actual gradient updates, checkpoint reload,
and detail rendering on the 12K coordinate lattice. They do not establish real
capture quality or full-12K GPU memory/latency.
