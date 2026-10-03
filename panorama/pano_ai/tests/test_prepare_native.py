import io
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image
from panorama.pano_ai.data.prepare_native import decode_dng, prepare


class NativePreparationTests(unittest.TestCase):
    def test_raw_decode_preserves_resolution_and_orientation(self):
        raw = MagicMock()
        raw.__enter__.return_value = raw
        module = SimpleNamespace(imread=MagicMock(return_value=raw), ColorSpace=SimpleNamespace(sRGB=1))
        with patch.dict(sys.modules, rawpy=module):
            decode_dng(Path('photo.dng'))
        options = raw.postprocess.call_args.kwargs
        self.assertFalse(options['half_size'])
        self.assertEqual(options['user_flip'], 0)
        self.assertEqual(options['output_bps'], 8)

    def test_full_size_jpeg_preparation_retains_raw_and_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images').mkdir()
            raw = root / 'images' / 'photo.DNG'
            raw.write_bytes(b'raw retained')
            Image.new('RGB', (64, 32)).save(root / 'Stitched.jpg')
            pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
            with patch('panorama.pano_ai.data.prepare_native.decode_dng', return_value=pixels):
                report = prepare(root, root / 'prepared', target_size=(64, 32))
            with Image.open(root / 'prepared/images/photo.jpg') as image:
                self.assertEqual(image.format, 'JPEG')
                self.assertEqual(image.size, (64, 48))
                from PIL.JpegImagePlugin import get_sampling
                self.assertEqual(get_sampling(image), 0)
                self.assertTrue(all(v == 1 for table in image.quantization.values() for v in table))
            self.assertEqual(report['frames'][0]['filename'], 'images/photo.jpg')
            self.assertEqual(raw.read_bytes(), b'raw retained')
            self.assertEqual((root / 'prepared/Stitched.jpg').read_bytes(), (root / 'Stitched.jpg').read_bytes())
            self.assertFalse(report['ready_for_calibration_loading'])
            self.assertFalse((root / 'prepared/capture.json').exists())
            with self.assertRaises(FileExistsError):
                prepare(root, root / 'prepared', target_size=(64, 32))
            with self.assertRaises(ValueError):
                prepare(root, root / 'wrong_size')

    def test_existing_jpeg_needs_no_raw_decoder_and_is_copied_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images').mkdir()
            source = root / 'images/photo.jpeg'
            Image.new('RGB', (96, 64), (20, 80, 150)).save(source)
            Image.new('RGB', (64, 32)).save(root / 'Stitched.jpg')
            with patch('panorama.pano_ai.data.prepare_native.decode_dng', side_effect=AssertionError('RAW decoder called')):
                report = prepare(root, root / 'prepared', target_size=(64, 32))
            self.assertEqual(source.read_bytes(), (root / 'prepared/images/photo.jpg').read_bytes())
            self.assertEqual(report['frames'][0]['width'], 96)
            self.assertEqual(report['frames'][0]['height'], 64)

    def test_ptgui_jpeg_saves_rectified_pixels_matching_calibration(self):
        from panorama.pano_ai.data.ptgui_calibration import PTGuiCalibration
        from panorama.pano_ai.tests.test_ptgui_calibration import PTGuiCalibrationTests
        from panorama.stitching.capture import load_capture

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'images').mkdir()
            project = PTGuiCalibrationTests().project()
            project['project']['globallenses'][0]['shift']['params']['longside'] = .01
            pts_path = root / 'Panorama.pts'
            pts_path.write_text(json.dumps(project), encoding='utf-8')
            pixels = np.arange(48 * 64 * 3, dtype=np.uint8).reshape(48, 64, 3)
            for name in ('0', '1'):
                Image.fromarray(pixels).save(root / f'images/{name}.jpg')
            Image.new('RGB', (64, 32)).save(root / 'Stitched.jpg')
            with Image.open(root / 'images/0.jpg') as image:
                decoded = np.array(image.convert('RGB'))
            corrected, frame, zoom = PTGuiCalibration(pts_path).rectify('0', decoded)
            self.assertFalse(np.array_equal(corrected, decoded))
            expected = io.BytesIO()
            Image.fromarray(corrected).save(expected, format='JPEG', quality=100, subsampling=0)

            report = prepare(root, root / 'prepared', target_size=(64, 32), pts_path=pts_path)

            self.assertEqual((root / 'prepared/images/0.jpg').read_bytes(), expected.getvalue())
            self.assertEqual(report['frames'][0]['rectification_zoom'], zoom)
            capture = load_capture(root / 'prepared/capture.json')
            self.assertEqual(capture.frames[0].intrinsics, frame.intrinsics)
            self.assertTrue(report['ready_for_calibration_loading'])
