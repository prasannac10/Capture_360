import unittest

import cv2
import numpy as np

from .local_alignment import estimate_displacement, remap_local


class LocalAlignmentTests(unittest.TestCase):
    def test_inverse_map_scales_and_wraps(self):
        image = np.tile(np.arange(100, dtype=np.float32), (50, 1))
        field = np.zeros((25, 50, 2), np.float32)
        field[..., 0] = 2
        result = remap_local(image, field)
        np.testing.assert_array_equal(result, np.roll(image, -4, axis=1))

    def test_small_translation_is_recovered(self):
        rng = np.random.default_rng(3)
        gray = cv2.GaussianBlur(rng.integers(0, 256, (128, 256), np.uint8), (3, 3), .6)
        reference = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        source = np.roll(reference, 3, axis=1)
        valid = np.full(gray.shape, 255, np.uint8)
        field, report = estimate_displacement(source, reference, valid, valid, valid)
        self.assertTrue(report['accepted'])
        corrected = remap_local(source, field)
        self.assertLess(np.mean(np.abs(corrected.astype(float) - reference)),
                        .4 * np.mean(np.abs(source.astype(float) - reference)))

    def test_unsupported_overlap_stays_unchanged(self):
        image = np.full((128, 256, 3), 120, np.uint8)
        valid = np.full(image.shape[:2], 255, np.uint8)
        field, report = estimate_displacement(image, image, valid, valid, valid)
        self.assertFalse(report['accepted'])
        self.assertFalse(field.any())

    def test_invalid_overlap_cannot_drive_alignment(self):
        rng = np.random.default_rng(4)
        image = rng.integers(0, 256, (128, 256, 3), np.uint8)
        valid = np.full(image.shape[:2], 255, np.uint8)
        field, report = estimate_displacement(np.roll(image, 3, axis=1), image,
                                              valid, np.zeros_like(valid), valid)
        self.assertFalse(report['accepted'])
        self.assertFalse(field.any())


if __name__ == '__main__':
    unittest.main()
