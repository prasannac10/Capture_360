"""Geometry regression checks for ARCore projection."""

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from .pose_stitcher import stitch_pose_files


class PoseStitcherTest(unittest.TestCase):
    def test_identity_camera_orientation_and_black_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = np.zeros((100, 100, 3), np.uint8)
            frame[:50, :50] = (0, 0, 255)
            frame[:50, 50:] = (0, 255, 0)
            frame[50:, :50] = (255, 0, 0)
            cv2.imwrite(str(root / 'frame.png'), frame)
            record = dict(name='frame', m=np.eye(4).flatten(order='F').tolist(),
                          fx=50, fy=50, cx=50, cy=50, w=100, h=100)
            poses = root / 'poses.jsonl'
            poses.write_text(json.dumps(record), encoding='utf-8')
            result = stitch_pose_files([root / 'frame.png'], root / 'out.png', (360, 180), poses)
            np.testing.assert_array_equal(result[80, 170], (0, 0, 255))
            np.testing.assert_array_equal(result[80, 190], (0, 255, 0))
            np.testing.assert_array_equal(result[100, 170], (255, 0, 0))
            coverage = cv2.imread(str(root / 'out_coverage.png'), cv2.IMREAD_GRAYSCALE)
            self.assertEqual(coverage[100, 190], 255)  # Black source pixels are valid.
            self.assertEqual(coverage[90, 0], 0)  # Behind the camera.
            with self.assertRaisesRegex(ValueError, 'No ARCore pose'):
                stitch_pose_files([root / 'unknown.png'], root / 'out.png', (360, 180), poses)
            with self.assertRaisesRegex(ValueError, '2:1'):
                stitch_pose_files([root / 'frame.png'], root / 'out.png', (360, 200), poses)


if __name__ == '__main__':
    unittest.main()
