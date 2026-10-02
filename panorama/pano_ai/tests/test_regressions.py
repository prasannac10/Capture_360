"""CPU regressions for executable AI stitching and correction contracts."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from panorama.pano_ai.models.panorama_model import PanoramaModel
from panorama.pano_ai.models.decoder import MultiScalePanoramaDecoder
from panorama.pano_ai.models.restoration_backbone import RestorationUNet
from panorama.pano_ai.models.tile_spherical import equirect_dirs
from panorama.pano_ai.highres_correction import apply_tiled_model, HighResolutionCorrectionPipeline
from panorama.pano_ai.data.tile_dataset import VariableTilePanoramaDataset
from panorama.pano_ai.train.panorama_loss import PanoramaLoss
from panorama.pano_ai.utils.ema import EMA

torch.set_num_threads(1)


class ModelTests(unittest.TestCase):
    def test_uneven_tile_batches_forward_and_backward(self):
        model = PanoramaModel(feature_dim=8, pano_feature_size=(8, 16), output_size=(16, 32),
                              pretrained=False, attention_heads=2, attention_layers=1, output_tile=128)
        tiles = torch.rand(3, 3, 32, 32)
        def batches():
            yield 0, tiles, torch.tensor([[0., 0.], [16., 0.], [32., 0.]]), torch.full((3, 2), 32.)
        out = model.forward_scene(batches, torch.tensor([[64., 32.]]),
                                  torch.tensor([[25., 25., 32., 16., 1., 0.]]), torch.zeros(1, 3), 2)
        self.assertEqual(out.shape, (1, 3, 16, 32))
        out.mean().backward()
        grad = model.encoder.projection[0].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().sum()), 0)

    def test_decoder_coordinates_do_not_depend_on_tile_size(self):
        model = MultiScalePanoramaDecoder(4, output_size=(25, 47), output_tile=16, output_overlap=4)
        x = torch.randn(1, 4, 6, 11)
        a = model(x)
        model.output_tile = 64
        b = model(x)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)

    def test_longitude_wrap_equivariance(self):
        model = MultiScalePanoramaDecoder(4, output_size=(16, 32), output_tile=16, output_overlap=4)
        x = torch.randn(1, 4, 8, 16)
        torch.testing.assert_close(model(x.roll(3, -1)), model(x).roll(6, -1), atol=2e-6, rtol=2e-6)

    def test_projection_top_is_camera_up(self):
        rays = equirect_dirs(8, 16, torch.device('cpu'), torch.float32)
        self.assertLess(float(rays[0, 8, 1]), 0)
        self.assertGreater(float(rays[-1, 8, 1]), 0)
        self.assertFalse(torch.equal(rays[:, 0], rays[:, -1]))

    def test_restoration_preserves_unmasked_pixels(self):
        model = RestorationUNet(base_channels=8, mask_channels=1)
        x = torch.rand(1, 3, 32, 32)
        mask = torch.zeros(1, 1, 32, 32)
        mask[:, :, 10:20, 10:20] = 1
        out = model(x, mask)
        torch.testing.assert_close(out[:, :, :10], x[:, :, :10], atol=0, rtol=0)
        with self.assertRaisesRegex(ValueError, 'explicit mask'):
            model(x)

    def test_masked_tiling_and_validation(self):
        class White(nn.Module):
            def forward(self, x, mask):
                return torch.ones_like(x)
        image = np.full((20, 40, 3), 20, np.uint8)
        mask = np.zeros((20, 40), np.float32)
        mask[5:15, 5:15] = .5
        out = apply_tiled_model(image, White(), 32, 8, 'cpu', mask)
        np.testing.assert_array_equal(out[mask == 0], image[mask == 0])
        self.assertAlmostEqual(float(out[10, 10, 0]), 138, delta=1)
        with self.assertRaisesRegex(ValueError, 'overlap'):
            apply_tiled_model(image, White(), 32, 32, 'cpu', mask)
        with self.assertRaisesRegex(ValueError, 'match image'):
            apply_tiled_model(image, White(), 32, 8, 'cpu', mask[:10])

    def test_correction_sees_across_longitude_boundary(self):
        class HorizontalBlur(nn.Module):
            def forward(self, x):
                return (x.roll(1, -1) + x + x.roll(-1, -1)) / 3
        image = np.zeros((20, 40, 3), np.uint8)
        image[:, 0] = 255
        out = apply_tiled_model(image, HorizontalBlur(), 32, 8, 'cpu')
        self.assertGreater(int(out[10, -1, 0]), 60)

    def test_unsupported_stages_fail_before_work(self):
        with self.assertRaises(NotImplementedError):
            HighResolutionCorrectionPipeline({'advanced_corrections': {'toggles': {'seam_blending': True}}}, 'config.yaml')

    def test_ghost_pipeline_requires_mask(self):
        pipeline = HighResolutionCorrectionPipeline({'advanced_corrections': {
            'allow_untrained': True, 'toggles': {'ghost_removal': True}}}, 'config.yaml')
        with self.assertRaisesRegex(ValueError, 'explicit defect mask'):
            pipeline.run(np.zeros((32, 32, 3), np.uint8), None, 32, 8)

    def test_losses_follow_weights(self):
        pred = torch.full((1, 3, 12, 16), .2, requires_grad=True)
        target = torch.full_like(pred, .5)
        l1 = PanoramaLoss({'supervised': {'l1_weight': 2}})
        self.assertAlmostEqual(float(l1(pred, target).detach()), .6, places=5)
        ssim = PanoramaLoss({'supervised': {'l1_weight': 0, 'ssim_weight': 1}})
        self.assertAlmostEqual(float(ssim(target, target)), 0, places=5)
        loss = ssim(pred, target)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        with self.assertRaises(NotImplementedError):
            PanoramaLoss({'geometry': {'smoothness_weight': 1}})

    def test_ema_preserves_integer_buffers(self):
        model = nn.BatchNorm2d(2)
        ema = EMA(model, .5)
        model.num_batches_tracked.fill_(4)
        ema.update(model)
        self.assertEqual(ema.shadow['num_batches_tracked'].dtype, torch.int64)
        self.assertEqual(int(ema.shadow['num_batches_tracked']), 4)


class DatasetTests(unittest.TestCase):
    def test_arcore_filters_outputs_and_uses_profile_frame_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / 'scene'
            images = scene / 'images'
            images.mkdir(parents=True)
            records = []
            for i in range(42):
                name = f'frame_{i:02}'
                Image.new('RGB', (32, 32)).save(images / (name + '.jpg'))
                records.append(dict(name=name, m=np.eye(4).flatten(order='F').tolist(),
                                    fx=20, fy=21, cx=16, cy=15, w=32, h=32))
            (images / 'ar_poses.jsonl').write_text('\n'.join(map(json.dumps, records)))
            Image.new('RGB', (64, 32)).save(images / 'classical_panorama.jpg')
            config = {'profiles': {'test': {'width': 32, 'height': 32, 'projection': 'pinhole',
                                           'min_frames': 4, 'max_frames': 60}}}
            sample = VariableTilePanoramaDataset(root, False, 32, 8, input_config=config)[0]
            self.assertEqual(len(sample['frame_paths']), 42)
            torch.testing.assert_close(sample['poses'], torch.eye(3).expand(42, 3, 3))
            torch.testing.assert_close(sample['camera_params'][0], torch.tensor([20., 21., 16., 15., 1., 0.]))

    def test_named_legacy_frames_keep_pose_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / 'scene'
            images = scene / 'images'
            images.mkdir(parents=True)
            for name in ('b.jpg', 'a.jpg', 'arbitrary_output.jpg'):
                Image.new('RGB', (32, 32)).save(images / name)
            torch.save({'frame_names': ['b.jpg', 'a.jpg'], 'poses': torch.tensor([[10., 0., 0.], [20., 0., 0.]])}, scene / 'poses.pt')
            (scene / 'camera.json').write_text(json.dumps({'projection': 'pinhole', 'horizontal_fov_deg': 60}))
            sample = VariableTilePanoramaDataset(root, False, 32, 8, min_frames=2)[0]
            self.assertEqual([Path(p).name for p in sample['frame_paths']], ['b.jpg', 'a.jpg'])


if __name__ == '__main__':
    unittest.main()
