"""Three synthetic scenes exercise real trainers, checkpoints and inference.

This is an execution test, not a model quality benchmark or real ground truth.
"""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from panorama.stitching.capture import CameraFrame, Capture, load_capture
from ..data.tile_dataset import VariableTilePanoramaDataset
from ..tiled_inference import run_tiled_inference
from .run_training import run, TASKS


def synthetic_panorama(seed):
    rng = np.random.default_rng(seed)
    coarse = rng.integers(25, 225, (16, 32, 3), dtype=np.uint8)
    canvas = cv2.resize(coarse, (1024, 512), interpolation=cv2.INTER_CUBIC)
    for i in range(8):
        x = int(rng.integers(0, 1000))
        cv2.rectangle(canvas, (x, 50), (min(x + 12, 1023), 460), (40, 180, 90), -1)
    # Make the environment periodic before rendering perspective captures.
    canvas[:, -16:] = canvas[:, :16]
    return canvas


def render_capture(folder, panorama):
    images = folder / 'images'
    images.mkdir(parents=True)
    frames = []
    axis = np.arange(64, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    camera = np.stack(((xx - 31.5) / 28, -(yy - 31.5) / 28, -np.ones_like(xx)), -1)
    camera /= np.linalg.norm(camera, axis=-1, keepdims=True)
    for i, (yaw, pitch) in enumerate(((0, 0), (90, 0), (180, 0), (270, 0), (0, 90), (0, -90))):
        y, p = np.deg2rad([yaw, pitch])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        rotation = ry @ rx
        world = camera @ rotation.T
        u = ((np.arctan2(world[..., 0], -world[..., 2]) / (2 * np.pi) + .5) * 1024 - .5) % 1024
        v = (.5 - np.arcsin(np.clip(world[..., 1], -1, 1)) / np.pi) * 512 - .5
        frame = cv2.remap(panorama, u.astype(np.float32), np.clip(v, 0, 511).astype(np.float32),
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        Image.fromarray(frame).save(images / f'{i}.png')
        frames.append(CameraFrame(str(i), f'images/{i}.png', 64, 64, [28, 28, 31.5, 31.5],
                                  rotation.tolist(), [0., 0., 0.]))
    Capture(folder, frames, 'synthetic_smoke_only').save(folder / 'capture.json')
    Image.fromarray(panorama).save(folder / 'panorama.png')


def prepare(root):
    data = root / 'dataset'
    for i in range(3):
        split = 'train' if i < 2 else 'val'
        name = f'synthetic_{i}'
        panorama = synthetic_panorama(100 + i)
        render_capture(data / 'panorama' / split / name, panorama)
        rgb = panorama.astype(np.float32) / 255
        one = np.ones((512, 1024, 1), np.float32)
        for task in ('alignment', 'blending'):
            folder = data / 'pairs' / task / split
            folder.mkdir(parents=True, exist_ok=True)
            common = dict(contract=np.array('geometric_pairs_v1'), reference=rgb,
                          reference_valid=one, source_valid=one, supervision_valid=one, confidence=one)
            if task == 'alignment':
                flow = np.zeros((512, 1024, 2), np.float32)
                flow[..., 0] = 1 + i
                common.update(source=np.roll(rgb, 1+i, axis=1), flow=flow)
            else:
                source, reference = rgb.copy(), rgb.copy()
                source[:, :512] *= .6
                reference[:, 512:] *= .6
                weight = np.zeros_like(one)
                weight[:, :512] = 1
                common.update(source=source, reference=reference, weight=weight)
            np.savez_compressed(folder / f'{name}.npz', **common)
        for stage in ('glare', 'nadir_zenith', 'ghost_removal', 'color'):
            folder = data / 'restoration' / split / name / 'stages' / stage
            folder.mkdir(parents=True, exist_ok=True)
            before = panorama.copy()
            mask = np.zeros((512, 1024), np.uint8)
            mask[240:272, 496:528] = 255
            if stage == 'color':
                before = np.clip(before.astype(np.float32) * .8, 0, 255).astype(np.uint8)
            elif stage == 'glare':
                before[mask > 0] = 255
            elif stage == 'ghost_removal':
                before[mask > 0] = np.roll(panorama, 8, axis=1)[mask > 0]
            else:
                before[mask > 0] = 0
            Image.fromarray(before).save(folder / 'before.png')
            Image.fromarray(panorama).save(folder / 'after.png')
            if stage != 'color':
                Image.fromarray(mask).save(folder / 'mask.png')
    pipeline_path = Path(__file__).resolve().parents[2] / 'stitching' / 'config.yaml'
    small_model = dict(feature_dim=8, tile_size=64, tile_overlap=16, pano_feature_height=16, pano_feature_width=32,
                       output_height=64, output_width=128, train_output_height=64, train_output_width=128,
                       encoder=dict(backbone='resnet18', pretrained=False, feature_stride=8), attention=dict(heads=2, layers=1))
    job = dict(dataset_root='dataset', output_dir='training', pipeline_config=str(pipeline_path), seed=42,
               cpu_threads=2, epochs=2, tasks=[task for task in TASKS if task != 'combined'],
               panorama=dict(model=small_model, training=dict(mixed_precision=False, tile_batch_size=3,
                             decoder_output_tile=128, lr=.001, use_ema=True, ema_decay=.9),
                             input=dict(min_frames=6, max_frames=6, profiles={'synthetic': dict(width=64, height=64,
                                        projection='pinhole', min_frames=6, max_frames=6)}),
                             loss={'supervised': {'l1_weight': 1., 'ssim_weight': .2, 'perceptual_weight': 0.}}),
               alignment=dict(channels=8), blending=dict(channels=8))
    for stage in ('glare', 'nadir_zenith', 'ghost_removal', 'color'):
        job[stage] = dict(input_size=[64, 64], base_channels=8, batch_size=1, lr=.001)
    (data / 'dataset_manifest.json').write_text(json.dumps(dict(version=1, label_source='synthetic_smoke_only',
        train_scenes=['synthetic_0', 'synthetic_1'], validation_scenes=['synthetic_2']), indent=2), encoding='utf-8')
    path = root / 'smoke_job.yaml'
    path.write_text(yaml.safe_dump(job), encoding='utf-8')
    return path


def verify_inference(root, report):
    config = yaml.safe_load((root / 'training' / 'panorama' / 'training.yaml').read_text())
    config['ai_pipeline']['mode'] = 'tiled_neural'
    config['inference'].update(checkpoint=report['tasks']['panorama']['checkpoint'], use_ema=True, output_tile=128)
    config['correction']['toggles'] = {}
    config['advanced_corrections']['toggles'] = {}
    scene = root / 'dataset' / 'panorama' / 'val' / 'synthetic_2'
    path = root / 'inference.yaml'
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    legacy = run_tiled_inference(scene, path, root / 'inference_decoder')
    metadata = json.loads((legacy / 'metadata.json').read_text())
    if metadata['weights'] != 'ema':
        raise RuntimeError('EMA checkpoint was not selected at inference')
    mask = np.zeros((64, 128), np.uint8)
    mask[24:40, 56:72] = 255
    Image.fromarray(mask).save(scene / 'smoke_mask.png')
    (scene / 'defects.json').write_text(json.dumps(dict(version=1, coordinates='equirectangular', size=[128, 64],
        masks={kind: 'smoke_mask.png' for kind in ('photographer', 'moving_objects', 'glare')})))
    config['ai_pipeline']['mode'] = 'source_preserving'
    for task in ('alignment', 'blending'):
        config['ai_pipeline']['learned_views'][task] = dict(enabled=True, checkpoint=report['tasks'][task]['checkpoint'])
    config['correction'].update(toggles={key: True for key in ('glare', 'nadir_zenith', 'color')},
                                checkpoints={key: report['tasks'][key]['checkpoint'] for key in ('glare', 'nadir_zenith', 'color')})
    config['advanced_corrections'].update(toggles={'ghost_removal': True},
                                         checkpoints={'ghost_removal': report['tasks']['ghost_removal']['checkpoint']})
    config['classical_pose'] = dict(refine_poses=False, recover_capture_headings=False, exposure_compensation=True,
                                    seam_blending=True, seam_width=256, blend_bands=2)
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    output = run_tiled_inference(scene, path, root / 'inference_source')
    raw = np.array(Image.open(output / 'initial_panorama.png'))
    glare = np.array(Image.open(output / '01_glare.png'))
    if not np.array_equal(raw[mask == 0], glare[mask == 0]):
        raise RuntimeError('Reloaded glare checkpoint changed protected pixels')
    report['inference'] = dict(decoder=str(legacy), source_preserving=str(output), ema_loaded=True,
                               glare_unmasked_pixels_preserved=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='New directory for synthetic fixtures, checkpoints and report')
    parser.add_argument('--validate-captures', nargs='*', default=[], help='Optional real captures: validate only, never fabricate their training labels')
    args = parser.parse_args()
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Choose a new output folder; existing smoke artifacts are preserved')
    root.mkdir(parents=True, exist_ok=True)
    job_path = prepare(root)
    report = run(job_path)
    for task, result in report['tasks'].items():
        if not result['parameter_change_l1'] > 0:
            raise RuntimeError(f'{task}: no optimizer update observed')
    verify_inference(root, report)
    report['real_capture_validation'] = []
    for path in args.validate_captures:
        path = Path(path).resolve()
        capture = load_capture(path)
        dataset = VariableTilePanoramaDataset(path.parent, False, max_frames=100)
        sample = dataset[dataset.scenes.index(path)]
        report['real_capture_validation'].append(dict(scene=str(path), frames=len(capture.frames),
                                                      dimensions=sample['image_size'].int().tolist(), used_for_training=False))
    report.update(status='passed', scenes=3, training_scenes=2, validation_scenes=1,
                  label_source='synthetic procedural environment and controlled corruptions',
                  quality_claim='None: execution smoke test, not real-scene convergence or Travvir parity')
    (root / 'smoke_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Smoke test passed: {root / "smoke_report.json"}')


if __name__ == '__main__':
    main()
