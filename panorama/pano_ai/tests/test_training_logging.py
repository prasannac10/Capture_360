import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from panorama.pano_ai.train.logging import TrainingLogger
from panorama.pano_ai.train.metrics import image_quality


class TrainingLoggingTests(unittest.TestCase):
    def test_accuracy_counts_pixels_not_individual_channels(self):
        target = torch.zeros(1, 3, 2, 2)
        pred = target.clone()
        pred[0, 0, 0, 0] = 9/255
        pred[0, :, 1, 1] = 7/255
        self.assertEqual(image_quality(pred, target)['pixel_accuracy_pct'], 75.)
        exact = image_quality(target, target)
        self.assertEqual(exact['pixel_accuracy_pct'], 100.)
        self.assertEqual(exact['psnr'], 99.)
        self.assertAlmostEqual(exact['ssim'], 1.)

    def test_jsonl_appends_runs_and_reports_actual_learning_rates(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()) as output:
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            first = TrainingLogger(tmp, 'combined', model, optimizer, {'batch_size': 1}, start_epoch=0)
            optimizer.param_groups[0]['lr'] = .0002
            first.epoch(1, optimizer, {'loss': .4, 'psnr': 20., 'ssim': .8, 'pixel_accuracy_pct': 60.}, {'loss': .5})
            second = TrainingLogger(tmp, 'combined', model, optimizer, {}, start_epoch=1)
            rows = [json.loads(line) for line in (Path(tmp)/'training_log.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]['parameters_trainable'], 3)
            self.assertEqual(rows[1]['learning_rates'], [.0002])
            self.assertNotEqual(first.run_id, second.run_id)
            self.assertEqual(rows[2]['start_epoch'], 1)
            self.assertIn('train_pixel_accuracy_pct=60.', output.getvalue())
            self.assertIn('train_ssim_global=', output.getvalue())


if __name__ == '__main__':
    unittest.main()
