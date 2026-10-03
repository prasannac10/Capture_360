import unittest
from unittest.mock import patch

import numpy as np

from .local_alignment import align_views, protect_view_centres


class FixedCentreTests(unittest.TestCase):
    def fixture(self):
        h, w = 64, 128
        x = np.arange(w)[None, :]
        qualities = [np.broadcast_to(np.maximum(0, 1 - abs(x - centre) / 60), (h, w)).astype(np.float32)
                     for centre in (32, 96)]
        masks = [np.full((h, w), 255, np.uint8) for _ in range(2)]
        fields = [np.ones((h, w, 2), np.float32) for _ in range(2)]
        return fields, masks, qualities

    def test_anchor_and_view_core_fixed_but_sides_can_move(self):
        fields, masks, qualities = self.fixture()
        result = protect_view_centres(fields, masks, qualities)
        self.assertFalse(result[0].any())
        self.assertFalse(result[1][:, 96].any())
        self.assertGreater(float(result[1][:, 20].max()), 0)
        self.assertTrue(np.all(result[1] <= fields[1]))

    def test_selected_anchor_is_not_estimated(self):
        fields, masks, qualities = self.fixture()
        images = [np.zeros((64, 128, 3), np.uint8) for _ in range(2)]
        with patch('panorama.pano_classical.local_alignment.estimate_displacement',
                   return_value=(fields[0], {'accepted': True})) as estimator:
            result, report = align_views(images, masks, qualities, reference_index=1)
        self.assertEqual(estimator.call_count, 1)
        self.assertFalse(result[1].any())
        self.assertEqual(report['reference_index'], 1)

    def test_invalid_reference(self):
        with self.assertRaises(ValueError):
            protect_view_centres(*self.fixture(), reference_index=3)
