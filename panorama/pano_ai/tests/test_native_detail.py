"""Neural detail preservation, winner ownership and native-scale training."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

from panorama.pano_ai.data.native_rgb import NativeRGBSource
from panorama.pano_ai.models.decoder import MultiScalePanoramaDecoder
from panorama.pano_ai.models.panorama_model import PanoramaModel
from panorama.pano_ai.models.tile_spherical import project_camera_pixels, panorama_contract
from panorama.pano_ai.train.panorama_loss import PanoramaLoss
from panorama.pano_ai.train_tiled import run_scene, main as train_main
from panorama.pano_ai.tiled_inference import run_tiled_inference
from panorama.stitching.capture import CameraFrame, Capture
from panorama.stitching.dispatcher import load_config
from panorama.stitching.profiles import validate_frame_set

torch.set_num_threads(1)


class NativeDetailTests(unittest.TestCase):
    def sample(self, root, colors=None):
        paths = []
        rng = np.random.default_rng(3)
        for index in range(2):
            array = (rng.integers(20, 220, (32, 32, 3), dtype=np.uint8) if colors is None
                     else np.broadcast_to(np.array(colors[index], np.uint8), (32, 32, 3)).copy())
            path = root / f'{index}.png'
            Image.fromarray(array).save(path)
            paths.append(str(path))
        return dict(frame_paths=paths, tile_specs=[[(0, 0, 32, 32)]] * 2,
                    image_size=torch.tensor([[32., 32.]] * 2),
                    camera_params=torch.tensor([[20., 20., 16., 16., 1., 0.]] * 2),
                    poses=torch.eye(3).repeat(2, 1, 1), exposure={'gains': [1., 1.]},
                    _tile_size=32, _overlap=8, _native_crop_size=16, _seam_crop_probability=0.)

    def test_region_projection_matches_full_and_longitude_wrap(self):
        size = torch.tensor([[[32., 32.]]])
        camera = torch.tensor([[[20., 20., 16., 16., 1., 0.]]])
        rotation = torch.eye(3)[None, None]
        full = project_camera_pixels(size, camera, rotation, 32, 64)
        crop = project_camera_pixels(size, camera, rotation, 32, 64, region=(5, 20, 8, 12))
        for a, b in zip(full, crop):
            torch.testing.assert_close(a[..., 5:13, 20:32], b)
        wrapped = project_camera_pixels(size, camera, rotation, 32, 64, region=(0, -2, 32, 4))
        for a, b in zip(full, wrapped):
            torch.testing.assert_close(torch.cat((a[..., -2:], a[..., :2]), -1), b, atol=2e-4, rtol=2e-5)

    def test_ownership_does_not_average_conflicting_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = NativeRGBSource(self.sample(Path(tmp), [(255, 0, 0), (0, 0, 255)]))
            rgb, valid, disagreement = source((0, 0, 32, 64), (32, 64), torch.device('cpu'))
            observed = valid[0, 0] > 0
            self.assertTrue(observed.any())
            torch.testing.assert_close(rgb[0, 0][observed], torch.ones_like(rgb[0, 0][observed]))
            self.assertFalse(rgb[0, 2].any())
            self.assertGreater(float(disagreement[0, 0][observed].mean()), .3)

    def test_native_skip_preserves_texture_and_learned_tile_continuity(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = NativeRGBSource(self.sample(Path(tmp)))
            decoder = MultiScalePanoramaDecoder(4, output_size=(32, 64), refinement_blocks=0,
                                                output_tile=16, output_overlap=4, detail_mode='rgb_residual')
            features = torch.randn(1, 4, 8, 16)
            base, _, _ = source((0, 0, 32, 64), (32, 64), features.device)
            with torch.no_grad():
                torch.testing.assert_close(decoder(features, source), base, atol=1e-6, rtol=1e-6)
                decoder.detail[-1].weight.normal_(0, .03)
                tiled = decoder(features, source)
                decoder.output_tile = 128
                full = decoder(features, source)
                crop = decoder(features, source, region=(7, 19, 16, 16))
            torch.testing.assert_close(tiled, full, atol=2e-6, rtol=2e-6)
            torch.testing.assert_close(crop, full[..., 7:23, 19:35], atol=2e-6, rtol=2e-6)

    def test_native_training_updates_head_and_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = self.sample(Path(tmp))
            base, valid, _ = NativeRGBSource(sample)((0, 0, 32, 64), (32, 64), torch.device('cpu'))
            sample['gt_panorama'] = (base[0] + .04 * valid[0]).clamp(0, 1)
            model = PanoramaModel(feature_dim=8, pano_feature_size=(8, 16), output_size=(32, 64),
                                  pretrained=False, attention_heads=2, attention_layers=1,
                                  output_tile=128, detail_mode='rgb_residual', checkpoint_encoder=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            criterion = PanoramaLoss({'supervised': {'l1_weight': 1., 'edge_weight': .2}})
            for _ in range(2):
                loss = run_scene(model, sample, torch.device('cpu'), 2, True, optimizer,
                                 criterion=criterion)
                self.assertTrue(np.isfinite(loss))
            self.assertGreater(float(model.decoder.detail[-1].weight.detach().abs().sum()), 0)
            grad = model.encoder.projection[0].weight.grad
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)
            sample['gt_panorama'] = torch.zeros(3, 16, 32)
            with self.assertRaisesRegex(ValueError, 'do not upscale'):
                run_scene(model, sample, torch.device('cpu'), 2, False, criterion=criterion)

    def test_detail_crop_uses_actual_12k_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = NativeRGBSource(self.sample(Path(tmp)))
            decoder = MultiScalePanoramaDecoder(4, output_size=(6000, 12000), refinement_blocks=0,
                                                output_tile=256, detail_mode='rgb_residual')
            features = torch.randn(1, 4, 8, 16)
            region = (2800, 5500, 128, 128)
            with torch.no_grad():
                actual = decoder(features, source, region)
                expected, _, _ = source(region, (6000, 12000), features.device)
            self.assertEqual(actual.shape, (1, 3, 128, 128))
            torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_edge_loss_penalizes_missing_texture(self):
        target = torch.zeros(1, 3, 16, 32)
        target[..., ::2] = 1
        blurred = torch.full_like(target, .5, requires_grad=True)
        loss = PanoramaLoss({'supervised': {'l1_weight': 0., 'edge_weight': 1.}})(blurred, target)
        self.assertGreater(float(loss.detach()), .4)
        loss.backward()
        self.assertGreater(float(blurred.grad.abs().sum()), 0)

    def test_device_profiles_and_architecture_contract(self):
        config = load_config()
        self.assertEqual(config['ai_pipeline']['mode'], 'tiled_neural')
        self.assertEqual(config['model']['detail']['mode'], 'rgb_residual')
        self.assertNotEqual(panorama_contract('features'), panorama_contract('rgb_residual'))
        for device in ('mobile', 'drone_still'):
            name, _ = validate_frame_set([(3500, 2100), (1920, 1080)], config['input'], device)
            self.assertEqual(name, device)
        self.assertEqual(validate_frame_set([(1920, 1080)], config['input'])[0], 'mobile_landscape')

    def test_checkpoint_training_reload_and_architecture_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val'):
                scene = root / split / 'scene'
                scene.mkdir(parents=True)
                sample = self.sample(scene)
                frames = [CameraFrame(str(i), f'{i}.png', 32, 32, [20, 20, 16, 16],
                                      np.eye(3).tolist(), [0, 0, 0]) for i in range(2)]
                Capture(scene, frames, 'calibrated').save(scene / 'capture.json')
                base, _, _ = NativeRGBSource(sample)((0, 0, 32, 64), (32, 64), torch.device('cpu'))
                Image.fromarray((base[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(scene / 'panorama.png')
            config = load_config()
            config.pop('_config_path')
            config['model'].update(feature_dim=8, pano_feature_height=8, pano_feature_width=16,
                                   tile_size=32, tile_overlap=8, output_height=32, output_width=64,
                                   train_output_height=32, train_output_width=64,
                                   encoder={'backbone': 'resnet18', 'pretrained': False},
                                   attention={'heads': 2, 'layers': 1})
            config['training'].update(training_data=str(root / 'train'), val_data=str(root / 'val'),
                                      model_path=str(root / 'weights'), native_crop_size=16,
                                      native_crops_per_scene=2, decoder_output_tile=128,
                                      tile_batch_size=2, mixed_precision=False)
            config['input'] = {'exposure_compensation': {'enabled': False}, 'profiles': {
                'test': dict(width=32, height=32, projection='pinhole', min_frames=2, max_frames=2)},
                'min_frames': 2, 'max_frames': 2}
            config['loss'] = {'supervised': {'l1_weight': 1., 'edge_weight': .2}}
            checkpoint = root / 'weights' / config['training']['best_model_name']
            config['inference'].update(checkpoint=str(checkpoint), output_tile=128, tile_batch_size=2)
            path = root / 'config.yaml'
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            train_main(['--config', str(path), '--epochs', '1'])
            state = torch.load(checkpoint, map_location='cpu', weights_only=False)
            self.assertEqual(state['contract'], panorama_contract('rgb_residual'))
            output = run_tiled_inference(root / 'val' / 'scene', path, root / 'output')
            with Image.open(output / 'final_panorama.png') as image:
                self.assertEqual(image.size, (64, 32))
            # Architecture mismatch is rejected before loading weights, even
            # if the general legacy-comparison flag is enabled.
            state['config']['model']['detail'] = {'mode': 'features'}
            torch.save(state, checkpoint)
            config['inference']['allow_legacy_checkpoint'] = True
            path.write_text(yaml.safe_dump(config), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'detail architecture differs'):
                run_tiled_inference(root / 'val' / 'scene', path, root / 'bad_output')


if __name__ == '__main__':
    unittest.main()
