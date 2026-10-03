import unittest

import numpy as np

from .shadow_removal import remove_shadows


class ShadowRemovalTests(unittest.TestCase):
    def test_lifts_masked_shadow_and_preserves_dark_objects_outside(self):
        image = np.full((128, 256, 3), 160, np.uint8)
        image[40:90, 100:150] = 60
        image[5:20, 5:20] = 0
        mask = np.zeros(image.shape[:2], np.float32)
        mask[40:90, 100:150] = 1
        result = remove_shadows(image, mask)
        self.assertTrue(np.array_equal(result[mask == 0], image[mask == 0]))
        self.assertGreater(float(result[55:75, 115:135].mean()), 110)

    def test_empty_mask_and_invalid_inputs(self):
        image = np.full((32, 64, 3), 80, np.uint8)
        self.assertTrue(np.array_equal(image, remove_shadows(image, np.zeros((32, 64)))))
        for mask in (np.ones((32, 64)), np.zeros((3, 4)), np.full((32, 64), np.nan)):
            with self.assertRaises(ValueError):
                remove_shadows(image, mask)

    def test_longitude_crossing_mask(self):
        image = np.full((128, 256, 3), 160, np.uint8)
        mask = np.zeros((128, 256), np.float32)
        mask[40:90, :15] = mask[40:90, -15:] = 1
        image[mask > 0] = 60
        result = remove_shadows(image, mask)
        self.assertTrue(np.array_equal(result[mask == 0], image[mask == 0]))
        self.assertGreater(float(result[50:80, :8].mean()), 90)
