import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
import torch
import yaml
from panorama.pano_ai.data.prepare_mobile import prepare, render, default_cameras, degrade
from panorama.pano_ai.data.paired_panorama import PairedPanoramaDataset
from panorama.pano_ai.tests.test_paired_panorama import make_pair
from panorama.pano_ai.train.train_restoration import main as train


class MobileAdaptationTests(unittest.TestCase):
    def test_render_central_ray_matches_forward_direction(self):
        panorama = np.broadcast_to(np.arange(128, dtype=np.uint8)[None, :, None], (64, 128, 3)).copy()
        camera = default_cameras(33, 33)[8]  # Equator yaw zero.
        result = render(panorama, camera)
        self.assertAlmostEqual(int(result[16, 16, 0]), 64, delta=1)

    def test_augmentation_is_seeded(self):
        image = np.full((32, 64, 3), 120, np.uint8)
        a, params = degrade(image, np.random.default_rng(2))
        b, other = degrade(image, np.random.default_rng(2))
        np.testing.assert_array_equal(a, b)
        self.assertEqual(params, other)

    def test_synthetic_data_rejected_as_real_mobile(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = make_pair(tmp)
            p = folder / 'pair.json'
            row = json.loads(p.read_text())
            row['domain'] = 'synthetic_mobile'
            p.write_text(json.dumps(row))
            with self.assertRaisesRegex(ValueError, 'domain=real_mobile'):
                PairedPanoramaDataset(tmp, (16, 16), required_domain='real_mobile')

    def test_physical_scene_variants_cannot_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val'):
                folder = make_pair(root / split, split)
                p = folder / 'pair.json'
                row = json.loads(p.read_text()); row['source_scene_id'] = 'same_location'
                p.write_text(json.dumps(row))
            cfg = root / 'config.yaml'; cfg.write_text('input_size: [16, 16]\n')
            with self.assertRaisesRegex(ValueError, 'same scene IDs'):
                train(['--stage', 'combined', '--data', str(root/'train'), '--val-data', str(root/'val'), '--config', str(cfg)])

    def test_generator_writes_trainable_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            Image.fromarray(np.random.default_rng(4).integers(20, 200, (64, 128, 3), dtype=np.uint8)).save(root/'reference.png')
            spec = dict(version=1, scenes=[dict(scene_id='room', panorama='reference.png', split='train',
                                               reviewed=True, projection='equirectangular')])
            (root/'refs.json').write_text(json.dumps(spec))
            report = prepare(root/'refs.json', root/'output', output_width=64, frame_width=32, frame_height=24)
            self.assertEqual(report['domain'], 'synthetic_mobile')
            ds = PairedPanoramaDataset(root/'output/restoration/train', (16, 16))
            self.assertEqual(ds.source_scene_ids, {'room'})
            self.assertEqual(ds[0]['input'].shape, (3, 16, 16))

    def test_mobile_validation_selects_checkpoint_and_reports_baseline(self):
        # Synthetic unit fixtures exercise the real-mobile manifest route, not a real quality evaluation.
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val', 'mobile'):
                folder = make_pair(root/split, split)
                p = folder/'pair.json'; row = json.loads(p.read_text())
                row['domain'] = 'real_mobile' if split == 'mobile' else 'dslr'
                row['provenance'] = 'synthetic_unit_fixture'
                p.write_text(json.dumps(row))
            cfg = dict(input_size=[16, 16], crops_per_scene=2, epochs=1, base_channels=8, batch_size=1)
            (root/'config.yaml').write_text(yaml.safe_dump(cfg))
            train(['--stage', 'combined', '--data', str(root/'train'), '--val-data', str(root/'val'),
                   '--mobile-val-data', str(root/'mobile'), '--config', str(root/'config.yaml'), '--out', str(root/'out')])
            state = torch.load(root/'out/combined_best.pt', weights_only=False)
            self.assertEqual(state['checkpoint_selection'], 'mobile')
            self.assertEqual(state['selection_loss'], state['mobile_metrics']['loss'])
            self.assertIn('longitude_join_error', state['mobile_metrics']['visual'])
            self.assertIsInstance(state['mobile_beats_baseline_l1'], bool)


if __name__ == '__main__':
    unittest.main()
