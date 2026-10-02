"""Scene settings reach the stitcher without affecting unrelated captures."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from panorama.stitching import dispatcher


class SceneOptionTests(unittest.TestCase):
    def run_scene(self, scene, **kwargs):
        config = dispatcher.load_config()
        frame = np.full((20, 40, 3), 100, np.uint8)
        profile = {'projection': 'pinhole', 'min_frames': 1, 'max_frames': 60}
        with (patch.object(dispatcher, 'load_config', return_value=config),
              patch.object(dispatcher, 'validate_frame_set', return_value=('test', profile)),
              patch('cv2.imread', return_value=frame), patch('cv2.imwrite', return_value=True),
              patch.object(Path, 'mkdir'), patch.object(Path, 'write_text'),
              patch('panorama.pano_classical.corrections.ClassicalCorrectionPipeline.run', return_value=(frame, {})),
              patch('panorama.pano_classical.opencv_stitcher.stitch_files', return_value=frame) as stitcher):
            dispatcher.stitch([Path(scene) / 'images' / 'a.jpg'], 'outputs/test.jpg',
                              engine='classical', **kwargs)
            return stitcher.call_args.kwargs['pose_options']

    def test_reviewed_scene_settings_are_forwarded(self):
        options = self.run_scene('scene_0004')
        self.assertEqual(options['seam_width'], 1024)
        self.assertEqual(options['blend_bands'], 5)
        self.assertFalse(options['local_alignment'])
        self.assertEqual(options['source_regions'][0]['source'], 'nadir')

    def test_other_scene_keeps_defaults(self):
        options = self.run_scene('scene_0005')
        self.assertEqual(options['seam_width'], 1024)
        self.assertEqual(options['blend_bands'], 5)

    def test_explicit_options_take_precedence(self):
        options = self.run_scene('scene_0004', pose_options={'seam_width': 512})
        self.assertEqual(options['seam_width'], 512)


if __name__ == '__main__':
    unittest.main()
