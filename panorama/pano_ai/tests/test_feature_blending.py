"""Regressions for camera-balanced, smoothly weighted neural feature fusion."""
import unittest

import torch

from panorama.pano_ai.models.tile_spherical import project_tile_features, equirect_dirs
from panorama.pano_ai.models.panorama_model import PanoramaModel


torch.set_num_threads(1)


class FeatureBlendingTests(unittest.TestCase):
    def project(self, features, xy, wh, sizes=None, cameras=None, **kwargs):
        n = features.shape[1]
        if sizes is None:
            sizes = torch.tensor([[[12., 8.]]]).expand(1, n, 2)
        if cameras is None:
            cameras = torch.tensor([[[6., 6., 5.5, 3.5, 1., 90.]]]).expand(1, n, 6)
        rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(1, n, 3, 3)
        return project_tile_features(features, xy, wh, sizes, cameras, rotations,
                                     64, 256, feature_stride=1, **kwargs)

    def test_extra_overlapping_tile_does_not_overweight_camera(self):
        features = torch.zeros(1, 2, 2, 1, 8, 8)
        features[:, 1] = 1
        xy = torch.tensor([[[[0., 0.], [4., 0.]], [[0., 0.], [99., 99.]]]])
        wh = torch.tensor([[[[8., 8.], [8., 8.]], [[12., 8.], [8., 8.]]]])
        out, weight = self.project(features, xy, wh)
        # Identical cameras have equal influence, even where A has two tiles.
        torch.testing.assert_close(out[weight > 0], torch.full_like(out[weight > 0], .5))

    def test_constant_features_preserved_at_edges_and_gradients_finite(self):
        features = torch.full((1, 1, 1, 1, 8, 8), .7, requires_grad=True)
        xy = torch.zeros(1, 1, 1, 2)
        wh = torch.tensor([[[[12., 8.]]]])
        out, weight = self.project(features, xy, wh)
        torch.testing.assert_close(out[weight > 0], torch.full_like(out[weight > 0], .7))
        self.assertTrue((out[weight == 0] == 0).all())
        out.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)

    def test_tile_transition_is_smooth(self):
        features = torch.zeros(1, 1, 2, 1, 8, 8)
        features[:, :, 1] = 1
        xy = torch.tensor([[[[0., 0.], [4., 0.]]]])
        wh = torch.full((1, 1, 2, 2), 8.)
        out, weight = self.project(features, xy, wh)
        ray = equirect_dirs(64, 256, 'cpu', torch.float32)[32]
        u = 6 * ray[:, 0] / ray[:, 2].clamp_min(1e-6) + 5.5
        select = (ray[:, 2] > 0) & (u > 3) & (u < 9)
        values = out[0, 0, 32, select]
        self.assertGreater(values.max().item(), .99)
        self.assertLess(values.min().item(), .01)
        self.assertLess(values.diff().abs().max().item(), .15)

    def test_camera_edges_feather_and_fisheye_remains_finite(self):
        features = torch.ones(1, 1, 1, 1, 8, 8)
        xy = torch.zeros(1, 1, 1, 2)
        wh = torch.tensor([[[[12., 8.]]]])
        for projection in (0., 1.):
            cameras = torch.tensor([[[4., 4., 5.5, 3.5, projection, 180.]]])
            out, weight = self.project(features, xy, wh, cameras=cameras)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(((weight > 0) & (weight < .5)).any())
            self.assertTrue((weight == 0).any())
            torch.testing.assert_close(out[weight > 0], torch.ones_like(out[weight > 0]))

    def test_model_result_independent_of_stream_chunk_size(self):
        torch.manual_seed(11)
        model = PanoramaModel(feature_dim=8, pano_feature_size=(8, 16), output_size=(16, 32),
                              pretrained=False, attention_heads=2, attention_layers=1,
                              output_tile=128).eval()
        tiles = torch.rand(2, 3, 3, 32, 32)
        xy = torch.tensor([[0., 0.], [16., 0.], [32., 0.]])
        wh = torch.full((3, 2), 32.)
        def factory(chunk):
            def batches():
                for fi in range(2):
                    for start in range(0, 3, chunk):
                        yield fi, tiles[fi, start:start+chunk], xy[start:start+chunk], wh[start:start+chunk]
            return batches
        sizes = torch.tensor([[64., 32.], [64., 32.]])
        cameras = torch.tensor([[25., 25., 32., 16., 1., 0.]]).expand(2, 6)
        with torch.no_grad():
            a = model.forward_scene(factory(3), sizes, cameras, torch.zeros(2, 3), 3)
            b = model.forward_scene(factory(1), sizes, cameras, torch.zeros(2, 3), 1)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-6)
        # Training must backpropagate through multiple camera accumulations too.
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = model.forward_scene(factory(2), sizes, cameras, torch.zeros(2, 3), 2)
        out.mean().backward()
        grad = model.encoder.projection[0].weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
