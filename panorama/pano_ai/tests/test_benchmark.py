"""CPU execution checks for the GPU benchmark instrumentation/report."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from panorama.stitching.capture import CameraFrame, Capture
from panorama.pano_ai.train.benchmark import main
from panorama.pano_ai.train.compare_benchmarks import compare

torch.set_num_threads(1)


class BenchmarkTests(unittest.TestCase):
    def test_real_steps_and_named_profiler_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / 'data' / 'scene'
            scene.mkdir(parents=True)
            frames = []
            for i in range(2):
                Image.new('RGB', (32, 32), (100, 120, 140)).save(scene / f'{i}.png')
                frames.append(CameraFrame(str(i), f'{i}.png', 32, 32, [20, 20, 16, 16],
                                          np.eye(3).tolist(), [0, 0, 0]))
            Capture(scene, frames, 'calibrated').save(scene / 'capture.json')
            Image.new('RGB', (64, 32), (110, 130, 150)).save(scene / 'panorama.png')
            config = dict(model=dict(feature_dim=8, pano_feature_height=8, pano_feature_width=16,
                                    train_output_height=32, train_output_width=64, tile_size=32,
                                    tile_overlap=8, encoder=dict(backbone='resnet18', pretrained=False),
                                    attention=dict(heads=2, layers=1), detail=dict(mode='rgb_residual')),
                          training=dict(lr=.001, weight_decay=.0001, tile_batch_size=2,
                                        decoder_output_tile=128, native_crop_size=16,
                                        checkpoint_encoder=True, mixed_precision=False),
                          input=dict(exposure_compensation=dict(enabled=False), profiles={
                              'test': dict(width=32, height=32, projection='pinhole', min_frames=2, max_frames=2)}),
                          loss=dict(supervised=dict(l1_weight=1., edge_weight=.2)))
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            report = main(['--scene', str(scene), '--config', str(path), '--out', str(root / 'out'),
                           '--warmup', '1', '--steps', '2', '--allow-cpu'])
            self.assertEqual(report['measured_steps'], 2)
            self.assertEqual(len(report['scene_sha256']), 64)
            self.assertGreater(report['mean_step_seconds'], 0)
            for name in ('forward', 'backward', 'source_tile_loading', 'feature_projection',
                         'native_rgb_loading_projection', 'encoder', 'decoder'):
                self.assertIn('bench.' + name, report['profiler_stages'])
            self.assertTrue((root / 'out' / 'trace.json').exists())
            stored = json.loads((root / 'out' / 'report.json').read_text())
            self.assertEqual(stored['device'], 'cpu')
            self.assertIsNone(stored['peak_allocated_bytes'])
            comparison = compare(report, dict(report, mean_step_seconds=report['mean_step_seconds'] / 2))
            self.assertEqual(comparison['measured_step_speedup'], 2)
            with self.assertRaisesRegex(ValueError, 'scene_sha256'):
                compare(report, dict(report, scene_sha256='different scene'))
            with self.assertRaisesRegex(ValueError, 'empty/new'):
                main(['--scene', str(scene), '--config', str(path), '--out', str(root / 'out'), '--allow-cpu'])


if __name__ == '__main__':
    unittest.main()
