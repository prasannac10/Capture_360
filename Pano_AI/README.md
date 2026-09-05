# Pano_AI

`Pano_AI/` is an independent Python ML project; the Android `app/` is only a consumer/export target.

## Dual-camera design

The panorama model supports **variable N** views rather than assuming six frames:
- fisheye camera: typically 4-8 views for 360 degrees
- phone/perspective camera: typically 20-30 views
- supported runtime range: 4-30 views

Every scene has `images/*.png` and `poses.pt`. For supervised training it also has `panorama.png`. An optional `camera.json` identifies the projection. Example phone metadata:

```json
{"projection":"pinhole","image_width":4000,"image_height":3000,"fx":2100,"fy":2100,"cx":2000,"cy":1500}
```

If `camera.json` is absent, the scene defaults to 180-degree equidistant fisheye. Phone scenes should provide calibrated OpenCV intrinsics; a guessed FOV is acceptable only for early experiments.

## Model

`ImageEncoder -> SetAggregator -> differentiable Spherical Fusion -> PanoramaDecoder`.
The set aggregator is permutation invariant and spherical fusion accepts arbitrary N. Projection metadata selects either 180-degree fisheye or pinhole geometry. The projection implementation is vectorized over N so ONNX can expose a dynamic `num_frames` axis.

## Training

Train the geometric/learned baseline with `python train.py --config config.yaml`. The loader pads different N values within a batch and supplies a frame mask and camera parameters. Keep fisheye and phone scenes in the same training set only after camera metadata is correct; stratified validation by camera type is recommended.

## ONNX / Android

`export.py` exports dynamic N with inputs `images`, `rotations`, `frame_mask`, and `camera_params`. Android accepts 4-30 captured frames and passes the selected camera profile. The output is RGB `[1,3,256,512]` in `[0,1]`.

The learned checkpoint and ONNX artifact are intentionally not committed until real training data is supplied.
