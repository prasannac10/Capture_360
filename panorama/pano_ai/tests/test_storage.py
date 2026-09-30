import tempfile
import unittest
from pathlib import Path
from panorama.pano_ai.data.storage import materialize_dataset


class FakeS3:
    def __init__(self):
        self.objects = {'dataset/train/scene/capture.json': b'{}', 'dataset/val/scene/image.png': b'image'}
        self.downloads = 0

    def get_paginator(self, name):
        assert name == 'list_objects_v2'
        return self

    def paginate(self, **kwargs):
        for key, content in self.objects.items():
            yield {'Contents': [{'Key': key, 'Size': len(content), 'ETag': repr(content)}]}

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key])
        self.downloads += 1

    def head_object(self, Bucket, Key):
        return {'ETag': repr(self.objects[Key])}


class StorageTests(unittest.TestCase):
    def test_local_root_relative_to_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'data').mkdir()
            self.assertEqual(materialize_dataset('data', root / 'cache', root), root / 'data')

    def test_s3_preserves_layout_and_updates_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeS3()
            first = materialize_dataset('s3://bucket/dataset/', tmp, client=client)
            self.assertEqual((first / 'train/scene/capture.json').read_bytes(), b'{}')
            again = materialize_dataset('s3://bucket/dataset/', tmp, client=client)
            self.assertEqual(first, again)
            self.assertEqual(client.downloads, 2)
            client.objects['dataset/train/scene/capture.json'] = b'{"version": 1}'
            changed = materialize_dataset('s3://bucket/dataset/', tmp, client=client)
            self.assertNotEqual(first, changed)
            self.assertEqual((first / 'train/scene/capture.json').read_bytes(), b'{}')

    def test_s3_rejects_path_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = FakeS3()
            client.objects = {'dataset/../../escape.txt': b'bad'}
            with self.assertRaisesRegex(ValueError, 'Unsafe dataset key'):
                materialize_dataset('s3://bucket/dataset/', tmp, client=client)
            self.assertEqual(client.downloads, 0)


if __name__ == '__main__':
    unittest.main()
