import json
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image
from panorama.stitching.capture import Capture, CameraFrame, CV_BASIS, load_capture
from panorama.stitching.defects import load_defect_masks
from panorama.stitching.warp_guard import guard_warp, sample
from panorama.stitching.visual_regression import metrics
from panorama.pano_ai.models.pairwise import PairwiseHead, inverse_warp
from panorama.pano_ai.train.train_pairwise import task_loss
from panorama.pano_ai.train.restoration import _predict
from panorama.pano_ai.models.restoration_backbone import RestorationUNet
from panorama.pano_ai.data.correction_dataset import CorrectionPairDataset
from panorama.pano_ai.data.tile_dataset import VariableTilePanoramaDataset
from panorama.pano_classical.pose_stitcher import load_pose_records
from panorama.pano_classical.spherical_composition import remap_preserving_coverage, spherical_rays
from panorama.pano_ai.models.tile_spherical import equirect_dirs

torch.set_num_threads(1)


class Contracts(unittest.TestCase):
    def test_shared_capture_roundtrip_and_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / 'scene'
            scene.mkdir()
            Image.new('RGB', (32, 32)).save(scene / 'different_filename.png')
            rotation = [[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]
            frame = CameraFrame('camera_id', 'different_filename.png', 32, 32, [20, 21, 16, 15],
                                rotation, [1, 2, 3], depth={'path': 'depth.npy', 'format': 'npy_meters', 'units': 'metres'})
            Capture(scene, [frame], 'calibrated').save(scene / 'capture.json')
            capture = load_capture(scene)
            self.assertEqual(capture.frames[0].translation, [1, 2, 3])
            self.assertEqual(capture.frames[0].depth['format'], 'npy_meters')
            record = load_pose_records(scene / 'capture.json')['different_filename']
            np.testing.assert_allclose(np.array(record['m']).reshape(4, 4, order='F')[:3, :3], rotation)
            data = VariableTilePanoramaDataset(scene.parent, False, 32, 8, min_frames=1)[0]
            np.testing.assert_allclose(data['poses'][0], CV_BASIS @ rotation @ CV_BASIS)
            np.testing.assert_allclose(data['translations'][0], [1, -2, -3])
        rays = equirect_dirs(16, 32, torch.device('cpu'), torch.float32).numpy()
        np.testing.assert_allclose(rays, spherical_rays(32) @ CV_BASIS, atol=5e-7)

    def test_defects_do_not_treat_valid_poles_as_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            coverage = np.full((16, 32), 255, np.uint8)
            coverage[7, 8] = 0
            masks = load_defect_masks(tmp, coverage.shape, coverage)
            self.assertEqual(set(masks), {'missing_coverage'})
            self.assertEqual(masks['missing_coverage'].sum(), 1)
            self.assertEqual(masks['missing_coverage'][0].sum(), 0)

    def test_unreliable_warp_is_exact_geometric_fallback(self):
        rng = np.random.default_rng(4)
        image = rng.integers(0, 255, (32, 64, 3), dtype=np.uint8)
        valid = np.ones((32, 64), np.uint8)
        field = np.ones((32, 64, 2), np.float32)
        for confidence, reverse in ((np.zeros((32, 64)), -field), (np.ones((32, 64)), field * 4)):
            flow, report = guard_warp(image, image, valid, valid, field, confidence, reverse)
            self.assertFalse(report['accepted'])
            np.testing.assert_array_equal(flow, 0)
            np.testing.assert_array_equal(sample(image, flow), image)

    def test_flow_preserves_original_coverage(self):
        image = np.full((16, 32, 3), 80, np.uint8)
        mask = np.zeros((16, 32), np.uint8)
        mask[:, :16] = 255
        flow = np.zeros((16, 32, 2), np.float32)
        flow[..., 0] = 4
        result, coverage, _ = remap_preserving_coverage(image, mask, mask.astype(np.float32), flow)
        self.assertTrue(np.all(coverage[mask > 0] > 0))
        np.testing.assert_array_equal(result[mask > 0], image[mask > 0])

    def test_reliable_translation_is_accepted(self):
        image = np.random.default_rng(4).integers(0, 255, (64, 128, 3), dtype=np.uint8)
        field = np.zeros((64, 128, 2), np.float32)
        field[..., 0] = 1
        valid = np.ones((64, 128), np.uint8)
        flow, report = guard_warp(image, np.roll(image, 1, axis=1), valid, valid,
                                  field, valid.astype(np.float32), -field)
        self.assertTrue(report['accepted'], report)
        self.assertLess(report['error_after'], report['error_before'])
        self.assertGreater(float(flow[..., 0].max()), .9)

    def test_fisheye_refinement_uses_equidistant_rays(self):
        from panorama.pano_classical.pose_refinement import pixel_rays
        record = dict(fx=100, fy=100, cx=0, cy=0, projection='fisheye_180')
        rays = pixel_rays(np.array([[0, 0], [100 * np.pi / 2, 0]]), record)
        np.testing.assert_allclose(rays, [[0, 0, -1], [1, 0, 0]], atol=1e-7)

    def test_auxiliary_model_output_shapes(self):
        from panorama.pano_ai.models.advanced_corrections import OverlapDetectionUNet, ParallaxCorrectionDetector
        x = torch.rand(1, 3, 32, 64)
        self.assertEqual(OverlapDetectionUNet(8)(x, x)['confidence'].shape, (1, 1))
        self.assertEqual(ParallaxCorrectionDetector(8)(x, x, torch.ones(1, 1, 32, 64)).shape, (1, 2, 32, 64))

    def test_large_output_coordinates_survive_mixed_precision(self):
        from panorama.pano_ai.models.decoder import MultiScalePanoramaDecoder
        model = MultiScalePanoramaDecoder(4, output_size=(8, 4098), refinement_blocks=0,
                                           output_tile=256, output_overlap=0)
        x = torch.rand(1, 4, 4, 64)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            a = model(x)
            model.output_tile = 512
            b = model(x)
        torch.testing.assert_close(a, b, rtol=0, atol=.004)

    def test_training_and_inference_inverse_sampling_agree(self):
        image = np.random.default_rng(3).random((16, 32, 3)).astype(np.float32)
        flow = np.zeros((16, 32, 2), np.float32)
        flow[..., 0], flow[..., 1] = 2.5, -.5
        result = inverse_warp(torch.from_numpy(image).permute(2, 0, 1)[None],
                              torch.from_numpy(flow).permute(2, 0, 1)[None])
        np.testing.assert_allclose(result[0].permute(1, 2, 0).numpy(), sample(image, flow), atol=2e-6)

    def test_separate_tasks_backward(self):
        for task in ('alignment', 'blending'):
            model = PairwiseHead(task, channels=4)
            rgb, mask = torch.rand(1, 3, 16, 32), torch.ones(1, 1, 16, 32)
            batch = dict(reference=rgb, source=rgb, reference_valid=mask, source_valid=mask,
                         supervision_valid=mask, confidence=mask, flow=torch.zeros(1, 2, 16, 32), weight=mask)
            loss = task_loss(model, batch)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_restoration_training_needs_mask_and_preserves_pixels(self):
        model = RestorationUNet(base_channels=8, mask_channels=1)
        batch = {'input': torch.rand(1, 3, 16, 32)}
        with self.assertRaisesRegex(ValueError, 'explicit defect mask'):
            _predict(model, batch, 'cpu', 'nadir_zenith')
        batch['mask'] = torch.zeros(1, 1, 16, 32)
        batch['mask'][..., 5:10, 8:16] = .5
        result = _predict(model, batch, 'cpu', 'nadir_zenith')
        protected = (batch['mask'] == 0).expand_as(result)
        torch.testing.assert_close(result[protected], batch['input'][protected], rtol=0, atol=0)

    def test_native_restoration_crop_and_required_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'scene' / 'stages' / 'nadir_zenith'
            folder.mkdir(parents=True)
            image = np.random.default_rng(7).integers(0, 255, (32, 64, 3), dtype=np.uint8)
            Image.fromarray(image).save(folder / 'before.png')
            Image.fromarray(image).save(folder / 'after.png')
            ds = CorrectionPairDataset(tmp, 'nadir_zenith', (16, 32))
            with self.assertRaisesRegex(ValueError, 'mask required'):
                ds[0]
            mask = np.zeros((32, 64), np.uint8)
            mask[16, 0] = 255
            Image.fromarray(mask).save(folder / 'mask.png')
            item = ds[0]
            self.assertEqual(item['mask'].sum(), 1)
            torch.testing.assert_close(item['input'], item['target'], rtol=0, atol=0)
            self.assertEqual(item['input'].shape, (3, 16, 32))

    def test_visual_metrics_detect_blur_seam_and_unmasked_change(self):
        image = np.random.default_rng(9).integers(0, 255, (32, 64, 3), dtype=np.uint8)
        identity = metrics(image, image, image, np.zeros((32, 64)))
        self.assertTrue(all(value == 0 for value in identity.values()))
        blur = cv2.GaussianBlur(image, (5, 5), 1)
        result = metrics(blur, image, image, np.zeros((32, 64)))
        self.assertGreater(result['edge_error'], 0)
        self.assertGreater(result['texture_error'], 0)
        self.assertGreater(result['longitude_join_error'], 0)
        self.assertGreater(result['unmasked_changed_pixels'], 0)

    def test_source_pipeline_runs_without_learned_weights(self):
        from panorama.pano_ai.source_inference import run_source_inference
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / 'scene'
            scene.mkdir()
            frames = []
            for i in range(2):
                Image.new('RGB', (32, 32), (80, 100, 120)).save(scene / f'{i}.png')
                frames.append(CameraFrame(f'id{i}', f'{i}.png', 32, 32, [20, 20, 16, 16], np.eye(3).tolist(), [0, 0, 0]))
            Capture(scene, frames, 'calibrated').save(scene / 'capture.json')
            config = dict(model=dict(tile_size=32, tile_overlap=8, output_width=64, output_height=32),
                          input={'profiles': {'test': dict(width=32, height=32, projection='pinhole', min_frames=2, max_frames=2)}},
                          classical_pose=dict(refine_poses=False, recover_capture_headings=False, seam_blending=False),
                          ai_pipeline={}, inference={'output_dir': 'out'})
            out = run_source_inference(scene, config, root / 'config.yaml', root / 'out')
            initial, final = np.array(Image.open(out / 'initial_panorama.png')), np.array(Image.open(out / 'final_panorama.png'))
            np.testing.assert_array_equal(initial, np.array(Image.open(out / 'initial_panorama.tiff')))
            np.testing.assert_array_equal(final, np.array(Image.open(out / 'final_panorama.tiff')))
            np.testing.assert_array_equal(initial, final)
            report = json.loads((out / 'metadata.json').read_text())
            self.assertEqual(report['learned_tasks'], [])
            self.assertIn('missing_coverage', report['defects'])


if __name__ == '__main__':
    unittest.main()
