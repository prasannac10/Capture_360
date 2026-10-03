"""Unobserved reference content must not supervise stitching."""
import unittest
import numpy as np
import torch
from panorama.pano_ai.train.panorama_loss import PanoramaLoss
from panorama.pano_ai.train.metrics import image_quality
from panorama.pano_ai.data.exposure import apply_exposure

class CoverageSupervisionTests(unittest.TestCase):
    def test_uncovered_content_and_boundary_do_not_change_loss_or_gradients(self):
        torch.manual_seed(1)
        pred = torch.rand(1, 3, 12, 12, requires_grad=True)
        target = torch.rand_like(pred)
        mask = torch.ones(1, 1, 12, 12)
        mask[..., :4, :] = 0
        altered = target.clone()
        altered[..., :4, :] = 20
        loss = PanoramaLoss({'supervised': {'l1_weight': 1, 'ssim_weight': .2, 'edge_weight': .2}})
        a, b = loss(pred, target, mask), loss(pred, altered, mask)
        torch.testing.assert_close(a, b)
        a.backward()
        self.assertEqual(float(pred.grad[..., :4, :].abs().sum()), 0)
        self.assertEqual(image_quality(pred, target, mask), image_quality(pred, altered, mask))
        self.assertTrue(torch.isfinite(loss(pred, target, torch.zeros_like(mask))))

    def test_native_exposure_agrees_with_feature_preprocessing(self):
        import tempfile
        from pathlib import Path
        from panorama.pano_ai.data.native_rgb import NativeRGBSource
        from panorama.pano_ai.tests.test_native_detail import NativeDetailTests
        with tempfile.TemporaryDirectory() as tmp:
            sample = NativeDetailTests().sample(Path(tmp), [(80, 120, 180)] * 2)
            sample['exposure']['gains'] = [1.7, 1.7]
            source = NativeRGBSource(sample, coverage_size=(32, 64))
            rgb, coverage, _ = source((-2, -2, 36, 68), (32, 64), torch.device('cpu'))
            expected = apply_exposure(np.array([80, 120, 180], np.float32)/255, 1.7)
            observed = coverage[0, 0] > 0
            np.testing.assert_allclose(rgb[0, :, observed].mean(1).numpy(), expected, atol=2e-7)
            torch.testing.assert_close(source.coverage_map, coverage[0, 0, 2:-2, 2:-2].bool())
