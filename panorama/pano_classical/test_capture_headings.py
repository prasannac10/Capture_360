"""Checks for guided-capture heading jumps without inventing missing views."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

from .capture_headings import recover_capture_headings


class CaptureHeadingTests(unittest.TestCase):
    def setUp(self):
        counts = {'equator': 12, 'down': 10, 'up': 10, 'bottom': 4, 'top': 4}
        self.names = [f'{row}_{i:02}' for row, n in counts.items() for i in range(1, n + 1)]
        headings = [(i - 1) * 360 / n for row, n in counts.items() for i in range(1, n + 1)]
        self.rotations = Rotation.from_euler('y', -np.array(headings)[:, None], degrees=True).as_matrix()
        self.records = [{'ts': i} for i in range(len(self.names))]
        self.manifest = json.dumps({'source': 'arcore', 'slots': self.names})

    def recover(self, rotations):
        with patch.object(Path, 'is_file', return_value=True), patch.object(Path, 'read_text', return_value=self.manifest):
            return recover_capture_headings([Path(n + '.jpg') for n in self.names], self.records,
                                            rotations, 'capture-info.json')

    def test_consistent_poses_unchanged(self):
        recovered, report = self.recover(self.rotations)
        self.assertEqual(report['status'], 'consistent')
        np.testing.assert_array_equal(recovered, self.rotations)

    def test_coherent_tracking_jump_is_recovered(self):
        shifted = self.rotations.copy()
        indices = [2, 3, 4, 5, 6, 7, 14, 15]
        shifted[indices] = Rotation.from_euler('y', -86, degrees=True).as_matrix() @ shifted[indices]
        recovered, report = self.recover(shifted)
        self.assertEqual(len(report['corrected_frames']), len(indices))
        np.testing.assert_allclose(recovered, self.rotations, atol=1e-12)

    def test_single_bad_slot_cannot_establish_tracking_reset(self):
        shifted = self.rotations.copy()
        shifted[3] = Rotation.from_euler('y', -86, degrees=True).as_matrix() @ shifted[3]
        recovered, report = self.recover(shifted)
        self.assertEqual(report['status'], 'consistent')
        np.testing.assert_array_equal(recovered, shifted)


if __name__ == '__main__':
    unittest.main()
