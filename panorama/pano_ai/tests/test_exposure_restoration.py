"""Photometry, spatial correction, and seam-safe inference regressions."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
from torch import nn

from panorama.pano_ai.data.exposure import (solve_overlap_gains, apply_exposure,
                                           estimate_exposure, linear_to_srgb)
from panorama.pano_ai.data.tile_dataset import VariableTilePanoramaDataset, iter_tile_batches
from panorama.pano_ai.models.combined_restoration import (CombinedRestorationUNet,
    LegacyCombinedRestorationUNet, COMBINED_CONTRACT, LEGACY_COMBINED_CONTRACT)
from panorama.pano_ai.highres_correction import apply_tiled_model, HighResolutionCorrectionPipeline
from panorama.pano_ai.train.restoration_loss import combined_loss
from panorama.pano_ai.data.paired_panorama import PairedPanoramaDataset

torch.set_num_threads(1)


class ExposureTests(unittest.TestCase):
    def test_known_relative_exposures_and_isolated_frame(self):
        scene = np.random.default_rng(4).uniform(.05, .25, (16, 32))
        luminance = [scene * factor for factor in (.5, 1, 2, 1)]
        masks = [np.ones_like(scene, dtype=bool) for _ in range(3)] + [np.zeros_like(scene, dtype=bool)]
        gains, report = solve_overlap_gains(luminance, masks)
        np.testing.assert_allclose(gains, [2, 1, .5, 1], atol=1e-6)
        self.assertLess(report['mean_log_error_after'], 1e-6)
        self.assertGreater(report['mean_log_error_before'], .5)

    def test_bad_overlaps_and_gain_limits(self):
        ramp = np.linspace(.01, 1, 256).reshape(16, 16)
        gains, report = solve_overlap_gains([ramp, ramp[::-1]], [np.ones_like(ramp, bool)] * 2)
        self.assertEqual(gains, [1., 1.])
        self.assertEqual(report['overlap_pairs'], 0)
        gains, _ = solve_overlap_gains([ramp, ramp * 100], [np.ones_like(ramp, bool)] * 2)
        self.assertTrue(all(.5 <= g <= 2 for g in gains))
        gains, _ = solve_overlap_gains([ramp * 0, ramp], [np.ones_like(ramp, bool)] * 2)
        self.assertEqual(gains, [1., 1.])

    def test_linear_gain_preserves_color_ratios(self):
        from panorama.pano_ai.data.exposure import srgb_to_linear
        rgb = np.array([[[.1, .2, .3]]], np.float32)
        result = apply_exposure(rgb, 1.5)
        np.testing.assert_allclose(srgb_to_linear(result), srgb_to_linear(rgb) * 1.5, atol=1e-6)
        np.testing.assert_array_equal(apply_exposure(rgb, 1.), rgb)

    def test_calibrated_estimator_and_shared_tile_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / 'scene'
            images = scene / 'images'
            images.mkdir(parents=True)
            base = np.full((32, 64, 3), .12, np.float32)
            for index, factor in enumerate((.5, 2.)):
                Image.fromarray(np.rint(linear_to_srgb(base * factor) * 255).astype(np.uint8)).save(images / f'{index}.png')
            torch.save({'frame_names': ['0.png', '1.png'], 'poses': torch.zeros(2, 3)}, scene / 'poses.pt')
            (scene / 'camera.json').write_text(json.dumps({'projection': 'pinhole', 'horizontal_fov_deg': 90}))
            ds = VariableTilePanoramaDataset(tmp, False, 32, 8, min_frames=2,
                                            exposure_config={'preview_width': 128, 'min_overlap': 16})
            sample = ds[0]
            self.assertGreater(sample['exposure']['overlap_pairs'], 0)
            chunks = list(iter_tile_batches(sample, 32, 8, 4))
            self.assertLess(abs(float(chunks[0][1].mean() - chunks[1][1].mean())), .005)
            self.assertEqual(sample['exposure'], ds[0]['exposure'])
            self.assertEqual(estimate_exposure(sample, {'enabled': False})['gains'], [1., 1.])


class RestorationTests(unittest.TestCase):
    def test_local_edit_sampling_and_unbiased_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'scene'
            root.mkdir()
            before = np.full((32, 64, 3), 100, np.uint8)
            after = before.copy(); after[12:16, 30:34] = 200
            Image.fromarray(before).save(root / 'before.png')
            Image.fromarray(after).save(root / 'after.png')
            (root / 'pair.json').write_text(json.dumps(dict(scene_id='scene', before='before.png',
                after='after.png', projection='equirectangular', alignment_verified=True)))
            training = PairedPanoramaDataset(tmp, (8, 8), random_crops=True, changed_crop_probability=1.)
            for index in range(len(training)):
                item = training[index]
                self.assertGreater((item['input'] - item['target']).abs().sum().item(), 0)
            a = PairedPanoramaDataset(tmp, (8, 8), changed_crop_probability=0.)
            b = PairedPanoramaDataset(tmp, (8, 8), changed_crop_probability=1.)
            for index in range(len(a)):
                torch.testing.assert_close(a[index]['input'], b[index]['input'], rtol=0, atol=0)

    def test_stitching_encoder_keeps_tile_batch_statistics_fixed(self):
        from panorama.pano_ai.models.encoder import ImageEncoder
        encoder = ImageEncoder(8, pretrained=False).train()
        bn = encoder.backbone[1]
        before = bn.running_mean.clone()
        tile = torch.rand(1, 3, 32, 32)
        a = encoder(tile)
        b = encoder(torch.cat([tile, torch.ones_like(tile)]))[:1]
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(bn.running_mean, before, rtol=0, atol=0)

    def test_identity_initialization_and_no_tile_statistics(self):
        model = CombinedRestorationUNet(8)
        self.assertFalse(any(isinstance(layer, (nn.GroupNorm, nn.BatchNorm2d)) for layer in model.modules()))
        x = torch.rand(1, 3, 17, 25)
        torch.testing.assert_close(model(x), x, rtol=0, atol=0)
        loss = combined_loss(model(x), (x + .05).clamp(0, 1), x)
        loss.backward()
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_structural_loss_penalizes_blur_and_ghosting(self):
        target = torch.zeros(1, 3, 32, 64)
        target[..., 12:20, 20:40] = .8
        blurred = torch.nn.functional.avg_pool2d(target, 5, 1, 2)
        ghost = .5 * (target + target.roll(5, -1))
        exact = combined_loss(target, target, ghost).item()
        self.assertLess(exact, 1e-6)
        for damaged in (blurred, ghost):
            self.assertGreater(combined_loss(damaged, target, damaged).item(), .01)
        # Same pixel MAE, but alternating errors damage edges more than a flat offset.
        flat = torch.full_like(target, .1)
        stripe = flat.clone(); stripe[..., ::2] *= -1
        weights = dict(ssim=0., multiscale=0., changed_regions=0.)
        self.assertGreater(combined_loss(target+stripe, target, target, weights),
                           combined_loss(target+flat, target, target, weights))

    def test_tiled_correction_preserves_identity_and_pooling_phase(self):
        image = np.random.default_rng(4).integers(0, 256, (17, 35, 3), dtype=np.uint8)
        model = CombinedRestorationUNet(8).eval()
        np.testing.assert_array_equal(apply_tiled_model(image, model, 40, 8, 'cpu'), image)
        with torch.no_grad():
            model.head.weight.normal_(0, .1)
            model.head.bias.fill_(.02)
        a = apply_tiled_model(image, model, 24, 8, 'cpu')
        b = apply_tiled_model(image, model, 40, 8, 'cpu')
        self.assertGreater(np.abs(a.astype(float) - image).mean(), 1.)
        self.assertLessEqual(np.abs(a.astype(int) - b.astype(int)).max(), 1)

    def test_checkpoint_architecture_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for cls, contract in ((CombinedRestorationUNet, COMBINED_CONTRACT),
                                  (LegacyCombinedRestorationUNet, LEGACY_COMBINED_CONTRACT)):
                path = root / (contract + '.pt')
                torch.save(dict(task='combined', contract=contract, channels=8, model=cls(8).state_dict()), path)
                pipeline = HighResolutionCorrectionPipeline({'correction': {'toggles': {'combined': True},
                    'checkpoints': {'combined': str(path)}}}, root / 'config.yaml')
                self.assertIsInstance(pipeline.models['combined'], cls)


if __name__ == '__main__':
    unittest.main()
