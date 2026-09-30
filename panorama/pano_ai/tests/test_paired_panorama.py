import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from panorama.pano_ai.data.paired_panorama import PairedPanoramaDataset
from panorama.pano_ai.highres_correction import HighResolutionCorrectionPipeline


def make_pair(root, scene_id='scene', verified=True):
    folder = Path(root) / scene_id
    folder.mkdir(parents=True)
    image = np.broadcast_to(np.arange(64, dtype=np.uint8)[None, :, None], (32, 64, 3)).copy()
    Image.fromarray(image).save(folder / 'before.png')
    Image.fromarray(image + 10).save(folder / 'after.png')
    (folder / 'pair.json').write_text(json.dumps(dict(scene_id=scene_id, before='before.png', after='after.png',
        projection='equirectangular', alignment_verified=verified)))
    return folder


class PairedTests(unittest.TestCase):
    def test_native_paired_crops_wrap_and_need_no_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_pair(tmp)
            ds = PairedPanoramaDataset(tmp, (16, 16), 12)
            self.assertEqual(len(ds), 12)
            sample = ds[0]
            self.assertNotIn('mask', sample)
            np.testing.assert_array_equal(np.rint(sample['input'][0, 0].numpy() * 255), list(range(56, 64)) + list(range(8)))
            np.testing.assert_allclose(sample['target'] - sample['input'], 10/255, atol=1e-7)

    def test_unreviewed_pair_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_pair(tmp, verified=False)
            with self.assertRaisesRegex(ValueError, 'review alignment'):
                PairedPanoramaDataset(tmp, (16, 16))

    def test_mismatched_dimensions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = make_pair(tmp)
            Image.new('RGB', (32, 16)).save(folder / 'after.png')
            with self.assertRaisesRegex(ValueError, 'identical'):
                PairedPanoramaDataset(tmp, (16, 16))

    def test_scene_leakage_rejected_before_training(self):
        from panorama.pano_ai.train.train_restoration import main
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            make_pair(root / 'train')
            make_pair(root / 'val')
            (root / 'config.yaml').write_text('input_size: [16, 16]\n')
            with self.assertRaisesRegex(ValueError, 'same scene IDs'):
                main(['--stage', 'combined', '--data', str(root / 'train'), '--val-data', str(root / 'val'),
                      '--config', str(root / 'config.yaml')])

    def test_combined_cannot_stack_with_legacy_stages(self):
        with self.assertRaisesRegex(ValueError, 'enable it alone'):
            HighResolutionCorrectionPipeline({'correction': {'toggles': {'combined': True, 'color': True}}}, 'config.yaml')


if __name__ == '__main__':
    unittest.main()
