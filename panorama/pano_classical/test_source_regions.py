import unittest

import numpy as np

from .source_regions import preserve_source_regions


class SourceRegionTests(unittest.TestCase):
    def test_ownership_changes_only_observed_region(self):
        full = np.full((100, 200), 255, np.uint8)
        partial = full.copy()
        partial[:, 110:] = 0
        original = [np.zeros_like(full), full.copy()]
        region = {'source': 'a', 'polygon': [[.4, .4], [.6, .4], [.6, .6], [.4, .6]]}
        result, report = preserve_source_regions(original, [partial, full], ['a.jpg', 'b.jpg'], [region])
        self.assertEqual(result[0][50, 100], 255)
        self.assertEqual(result[1][50, 100], 0)
        self.assertEqual(result[1][50, 115], 255)
        self.assertEqual(result[1][5, 5], 255)
        self.assertFalse(original[0].any())
        self.assertLess(report[0]['observed_fraction'], 1)
        np.testing.assert_array_equal(result[0] | result[1], full)

    def test_unknown_source_fails(self):
        mask = np.ones((100, 200), np.uint8)
        with self.assertRaisesRegex(ValueError, 'not an input frame'):
            preserve_source_regions([mask], [mask], ['a.jpg'], [{'source': 'missing'}])


if __name__ == '__main__':
    unittest.main()
