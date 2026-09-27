"""Regression checks for source geometry, ownership and finishing safeguards."""

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from .corrections import ClassicalCorrectionPipeline
from .parallax_correction_cv import ParallaxCorrectionCV
from .pose_refinement import optimize_rotations, recover_unmatched_rotations
from .seam_blending_cv import GhostRemovalCV, SeamBlendingCV
from .spherical_composition import compose_sphere, exposure_gains, seam_masks
from .overlap_detector_cv import OverlapDetectorCV
from panorama.stitching.dispatcher import load_config


class CorrectionTests(unittest.TestCase):
    def test_signed_subpixel_displacements(self):
        points = np.float32([[16, 16], [48, 16], [16, 48], [48, 48]])
        values = np.tile(np.float32([-5.25, 2.5]), (4, 1))
        field = ParallaxCorrectionCV()._interpolate_parallax(points, values, 64, 64)
        np.testing.assert_allclose(field[16, 16], [-5.25, 2.5], atol=1e-5)
        self.assertLessEqual(field[..., 0].max(), 0)

    def test_homography_inverse_sampling(self):
        image = np.zeros((64, 64, 3), np.uint8)
        image[24:36, 20:30] = 255
        h = np.float32([[1, 0, 5], [0, 1, 0], [0, 0, 1]])
        corrector = ParallaxCorrectionCV()
        field = corrector.estimate_parallax_shift(image, image, h, [], [], [])
        result = corrector.apply_parallax_correction(image, field)
        np.testing.assert_array_equal(result[24:36, 25:35], 255)
        self.assertFalse(result[:, 20:25].any())

    def test_flow_aligns_source_to_reference(self):
        rng = np.random.default_rng(42)
        source = rng.integers(0, 256, (128, 256, 3), dtype=np.uint8)
        source = cv2.GaussianBlur(source, (5, 5), 0)
        reference = cv2.warpAffine(source, np.float32([[1, 0, 4], [0, 1, 0]]), (256, 128))
        corrected = ClassicalCorrectionPipeline._correct_parallax_flow(source, reference)
        roi = np.s_[16:-16, 16:-16]
        before = np.abs(source[roi].astype(float) - reference[roi]).mean()
        after = np.abs(corrected[roi].astype(float) - reference[roi]).mean()
        self.assertLess(after, before * 0.1)

    def test_exposure_offset_is_not_a_ghost(self):
        image = np.full((64, 128, 3), 100, np.uint8)
        ghosts = GhostRemovalCV()
        self.assertFalse(ghosts.detect_ghost_regions(image, image + 10, np.ones((64, 128), bool)).any())
        self.assertFalse(ghosts.detect_ghost_regions(image, image, np.zeros((64, 128), bool)).any())

    def test_ghost_mask_stays_in_overlap(self):
        a = np.full((64, 128, 3), 100, np.uint8)
        b = a.copy()
        b[20:40, 50:100] = 200
        valid = np.zeros(a.shape[:2], bool)
        valid[:, :80] = True
        result = GhostRemovalCV().detect_ghost_regions(a, b, valid)
        self.assertTrue(result[25, 60])
        self.assertFalse(result[:, 80:].any())

    def test_median_is_not_mean(self):
        images = [np.full((8, 8, 3), n, np.uint8) for n in (10, 20, 200)]
        result = GhostRemovalCV().remove_ghosts_median_filter(images, [np.zeros((8, 8), bool)] * 3)
        np.testing.assert_array_equal(result, 20)

    def test_seam_is_region_and_constant_ownership_is_exact(self):
        seam = SeamBlendingCV('feather')
        cost = np.ones((32, 128), np.float32)
        cost[:, 64] = 0
        mask = seam._compute_minimum_cost_seam(cost, np.ones(cost.shape, bool))
        self.assertTrue(mask[:, :64].all())
        self.assertFalse(mask[:, 65:].any())
        a = np.full((32, 128, 3), 40, np.uint8)
        b = np.full_like(a, 200)
        np.testing.assert_array_equal(seam.blend_images(a, b, np.zeros(cost.shape, bool)), b)
        np.testing.assert_array_equal(seam.blend_images(a, b, np.ones(cost.shape, bool)), a)

    def test_graph_cut_preserves_exclusive_coverage(self):
        a = np.full((32, 128, 3), 90, np.uint8)
        b = np.full_like(a, 110)
        ma = np.zeros(a.shape[:2], np.uint8)
        mb = ma.copy()
        ma[:, :90], mb[:, 40:] = 255, 255
        result = seam_masks([a, b], [ma, mb])
        self.assertTrue((result[0][:, :40] > 0).all())
        self.assertTrue((result[1][:, 90:] > 0).all())
        self.assertFalse(result[0][:, 90:].any())
        self.assertFalse(result[1][:, :40].any())

    def test_exposure_gains_balance_overlap(self):
        a = np.full((32, 64, 3), 100, np.uint8)
        b = np.full_like(a, 120)
        masks = [np.full(a.shape[:2], 255, np.uint8)] * 2
        gains = exposure_gains([a, b], masks)
        self.assertLess(abs(100 * gains[0] - 120 * gains[1]), 1)

    def test_no_unmasked_finishing_or_fake_applied_stages(self):
        cfg = load_config()
        for section in ('correction', 'advanced_corrections'):
            cfg[section]['toggles'] = dict.fromkeys(cfg[section]['toggles'], True)
        cfg['correction']['toggles']['color'] = False
        cfg['correction']['toggles']['sharpen'] = False
        cfg['classical_finishing'] = {'color': False, 'sharpen': False}
        image = np.full((64, 128, 3), 250, np.uint8)
        result, report = ClassicalCorrectionPipeline(cfg).run(image, [])
        np.testing.assert_array_equal(result, image)
        self.assertEqual(report['applied_stages'], [])
        self.assertEqual(len(report['skipped_stages']), 8)

    def test_inpaint_requires_small_explicit_mask(self):
        image = np.full((64, 128, 3), 100, np.uint8)
        mask = np.zeros(image.shape[:2], bool)
        mask[30:32, 60:62] = True
        damaged = image.copy()
        damaged[mask] = 255
        result = ClassicalCorrectionPipeline._inpaint_mask(damaged, mask)
        np.testing.assert_array_equal(result[~mask], damaged[~mask])
        with self.assertRaises(ValueError):
            ClassicalCorrectionPipeline._inpaint_mask(image, np.ones(mask.shape, bool))

    def test_rotation_refinement_recovers_relative_alignment(self):
        rng = np.random.default_rng(7)
        world = rng.normal(size=(60, 3))
        world /= np.linalg.norm(world, axis=1, keepdims=True)
        true = Rotation.from_euler('y', [[0], [30], [60]], degrees=True).as_matrix()
        initial = Rotation.from_euler('y', [[2], [28], [60]], degrees=True).as_matrix()
        edges = [(i, j, world @ true[i], world @ true[j]) for i, j in ((0, 1), (1, 2))]
        refined, report = optimize_rotations(initial, edges)
        self.assertEqual(report['status'], 'refined')
        self.assertLess(report['median_error_after_deg'], 0.1)
        # Disconnected camera remains unchanged.
        np.testing.assert_array_equal(optimize_rotations(initial, [])[0], initial)

    def test_composition_does_not_create_dark_frame_borders(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f'{i}.png' for i in range(2)]
            for path in paths:
                self.assertTrue(cv2.imwrite(str(path), np.full((120, 160, 3), 100, np.uint8)))
            records = [dict(w=160, h=120, fx=100, fy=100, cx=80, cy=60)] * 2
            rotations = Rotation.from_euler('y', [[-20], [20]], degrees=True).as_matrix()
            image, mask, _ = compose_sphere(paths, records, rotations, 256)
            self.assertGreater(np.count_nonzero(mask), 1000)
            self.assertLessEqual(np.abs(image[mask > 0].astype(float) - 100).max(), 2)
            self.assertFalse(image[mask == 0].any())

    def test_black_pixels_are_valid_overlap(self):
        image = np.zeros((20, 40, 3), np.uint8)
        _, mask = OverlapDetectorCV().detect_overlap_region(image, image, np.eye(3))
        self.assertTrue(mask.all())

    def test_large_pose_recovery_requires_two_agreeing_neighbors(self):
        rng = np.random.default_rng(12)
        world = np.column_stack((rng.uniform(-0.3, 0.3, 40), rng.uniform(-0.1, 0.1, 40), -np.ones(40)))
        world /= np.linalg.norm(world, axis=1, keepdims=True)
        true = Rotation.from_euler('y', [[0], [30], [60]], degrees=True).as_matrix()
        recorded = true.copy()
        recorded[2] = Rotation.from_euler('y', 77, degrees=True).as_matrix()
        edges = [(i, 2, world @ true[i], world @ true[2]) for i in (0, 1)]
        unchanged, report = recover_unmatched_rotations(recorded, edges[:1], {0, 1})
        self.assertEqual(report, [])
        np.testing.assert_array_equal(unchanged, recorded)
        recovered, report = recover_unmatched_rotations(recorded, edges, {0, 1})
        self.assertEqual(len(report), 1)
        np.testing.assert_allclose(recovered[2], true[2], atol=1e-8)


if __name__ == '__main__':
    unittest.main()
