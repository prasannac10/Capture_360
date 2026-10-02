"""Object-removal regression checks, including OpenCV's mutated clone mask."""

import unittest

import numpy as np

from .object_removal import remove_configured_objects
from .corrections import ClassicalCorrectionPipeline
from panorama.stitching.dispatcher import load_config


class ObjectRemovalTests(unittest.TestCase):
    def settings(self):
        return {'regions': [{'polygon': [[.40, .40], [.45, .40], [.45, .49], [.40, .49]],
                             'donor_offset': [.20, 0]}]}

    def image(self, scale=1):
        image = np.full((256 * scale, 512 * scale, 3), 120, np.uint8)
        image[110 * scale:120 * scale, 210 * scale:220 * scale] = 0
        return image

    def test_object_interior_is_replaced_and_outside_is_unchanged(self):
        for scale in (1, 2):
            with self.subTest(scale=scale):
                image = self.image(scale)
                result, mask, _ = remove_configured_objects(image, self.settings())
                self.assertGreater(result[115 * scale, 215 * scale].min(), 115)
                np.testing.assert_array_equal(result[mask == 0], image[mask == 0])
                self.assertEqual(image[115 * scale, 215 * scale].max(), 0)

    def test_bad_region_and_contaminated_donor_are_rejected(self):
        settings = self.settings()
        settings['regions'][0]['donor_offset'] = [0, 0]
        with self.assertRaisesRegex(ValueError, 'donor overlaps'):
            remove_configured_objects(self.image(), settings)
        settings = self.settings()
        settings['regions'][0]['polygon'][0] = [-.1, .4]
        with self.assertRaisesRegex(ValueError, 'normalized'):
            remove_configured_objects(self.image(), settings)

    def test_area_limit_is_enforced(self):
        settings = self.settings()
        settings['max_area_fraction'] = .001
        with self.assertRaisesRegex(ValueError, 'exceeding'):
            remove_configured_objects(self.image(), settings)

    def test_config_toggle_controls_pipeline_and_audit(self):
        config = load_config()
        config['classical_finishing'] = {'object_removal': True, 'color': False, 'sharpen': False}
        config['classical_parameters']['object_removal'] = self.settings()
        result, report = ClassicalCorrectionPipeline(config).run(self.image(), [])
        self.assertEqual(report['applied_stages'], ['object_removal'])
        self.assertFalse(report['object_removal']['automatic_detection'])
        self.assertGreater(result[115, 215].min(), 115)
        config['classical_finishing']['object_removal'] = False
        result, report = ClassicalCorrectionPipeline(config).run(self.image(), [])
        np.testing.assert_array_equal(result, self.image())
        self.assertEqual(report['applied_stages'], [])

    def test_regions_are_not_reused_for_other_scenes(self):
        config = load_config()
        config['classical_finishing'] = {'object_removal': True, 'color': False, 'sharpen': False}
        config['classical_parameters']['object_removal'] = {'scenes': {'known_scene': self.settings()}}
        pipeline = ClassicalCorrectionPipeline(config)
        result, report = pipeline.run(self.image(), [], scene_name='other_scene')
        np.testing.assert_array_equal(result, self.image())
        self.assertIn('object_removal', report['skipped_stages'])
        result, report = pipeline.run(self.image(), [], scene_name='known_scene')
        self.assertGreater(result[115, 215].min(), 115)
        self.assertEqual(report['object_removal']['scene'], 'known_scene')


if __name__ == '__main__':
    unittest.main()
