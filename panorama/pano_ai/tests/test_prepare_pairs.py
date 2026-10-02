import json
from pathlib import Path
import tempfile
import unittest
from PIL import Image
from panorama.pano_ai.data.prepare_pairs import prepare
from panorama.pano_ai.data.paired_panorama import PairedPanoramaDataset


class PairPreparationTests(unittest.TestCase):
    def fixture(self, root):
        Image.new('RGB', (64, 32), (80, 100, 120)).save(root/'before.png')
        Image.new('RGB', (64, 32), (100, 120, 140)).save(root/'after.png')
        row = dict(scene_id='a', source_scene_id='location', split='train', domain='dslr',
                   before='before.png', after='after.png', alignment_verified=True, projection='equirectangular')
        return row

    def test_packaging_is_portable_and_preserves_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); row=self.fixture(root)
            (root/'pairs.json').write_text(json.dumps(dict(version=1,pairs=[row])))
            report=prepare(root/'pairs.json',root/'bundle',16)
            ds=PairedPanoramaDataset(root/'bundle/restoration/train',(16,16))
            self.assertEqual(ds.scene_ids,{'a'})
            self.assertEqual(report['counts'],{'train':1})
            self.assertEqual((root/'before.png').read_bytes(),(root/'bundle/restoration/train/a/before.png').read_bytes())

    def test_review_is_required_and_failure_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row=self.fixture(root);row['alignment_verified']=False
            (root/'pairs.json').write_text(json.dumps(dict(version=1,pairs=[row])))
            with self.assertRaisesRegex(ValueError,'Review pair'):
                prepare(root/'pairs.json',root/'bundle',16)
            self.assertFalse((root/'bundle').exists())

    def test_rejects_scene_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);row=self.fixture(root)
            second=dict(row,scene_id='b',split='val')
            (root/'pairs.json').write_text(json.dumps(dict(version=1,pairs=[row,second])))
            with self.assertRaisesRegex(ValueError,'single split'):
                prepare(root/'pairs.json',root/'bundle',16)


if __name__=='__main__':
    unittest.main()
