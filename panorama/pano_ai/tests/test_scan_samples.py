import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import yaml

from panorama.pano_ai.data.scan_samples import scan
from panorama.pano_ai.data.prepare_pairs import prepare


class ScanSamplesTests(unittest.TestCase):
    def scene(self, root, year, location):
        scene = root / year / location / 'Scene_001'
        (scene / 'images').mkdir(parents=True)
        for name in ('Stitched.jpg', 'Edited.jpg'):
            Image.new('RGB', (64, 32)).save(scene / name)
        return scene

    def test_all_input_levels_and_location_grouping(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / 'Files'
            self.scene(root, '2026', 'A')
            self.scene(root, '2026', 'B')
            self.scene(root, '2025', 'A')
            for index, selected in enumerate((root, root / '2026', root / '2026/A')):
                out = base / f'setup_{index}'
                report = scan(selected, out)
                rows = json.loads((out / 'pairs.json').read_text())['pairs']
                self.assertEqual(len(rows), (3, 2, 1)[index])
                self.assertTrue(all(row['domain'] is None and not row['alignment_verified'] for row in rows))
                self.assertEqual(len({row['scene_id'] for row in rows}), len(rows))
                for group in {row['source_scene_id'] for row in rows}:
                    self.assertEqual(len({r['split'] for r in rows if r['source_scene_id'] == group}), 1)
                self.assertEqual(report['needs_independent_validation'], index == 2)
                job = yaml.safe_load((out / 'training_job.yaml').read_text())
                self.assertTrue(Path(job['pipeline_config']).is_file())

    def test_reviewed_manifest_packages_and_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            self.scene(base / 'Files', '2026', 'A')
            self.scene(base / 'Files', '2026', 'B')
            out = base / 'setup'
            scan(base / 'Files', out, 'drone')
            with self.assertRaisesRegex(ValueError, 'existing files'):
                scan(base / 'Files', out, 'drone')
            manifest = out / 'pairs.json'
            with self.assertRaisesRegex(ValueError, 'Review pair'):
                prepare(manifest, out / 'training_bundle', 16)
            data = json.loads(manifest.read_text())
            for row in data['pairs']:
                row['alignment_verified'] = True
            manifest.write_text(json.dumps(data))
            self.assertEqual(prepare(manifest, out / 'training_bundle', 16)['counts'], {'train': 1, 'val': 1})

    def test_incomplete_scene_fails_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            scene = self.scene(base / 'Files', '2026', 'A')
            (scene / 'Edited.jpg').unlink()
            with self.assertRaisesRegex(ValueError, 'Incomplete scenes'):
                scan(base / 'Files', base / 'out')
            self.assertFalse((base / 'out').exists())
