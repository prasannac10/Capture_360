"""Epoch-boundary resume equivalence on a small CPU restoration model."""
import random
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
import json
from panorama.pano_ai.train.train_restoration import main

class ResumeTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_training_and_rejects_mismatch(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val'):
                folder = root / split / split
                folder.mkdir(parents=True)
                rgb = np.random.default_rng(7).integers(30, 170, (32, 64, 3), dtype=np.uint8)
                Image.fromarray(rgb).save(folder / 'before.png')
                Image.fromarray(rgb + 15).save(folder / 'after.png')
                (folder / 'pair.json').write_text(json.dumps(dict(scene_id=split, before='before.png', after='after.png', alignment_verified=True, projection='equirectangular')))
            def train(name, epochs, resume=None, channels=8, finetune=None):
                random.seed(42); np.random.seed(42); torch.manual_seed(42)
                cfg = dict(epochs=epochs, input_size=[16,16], crops_per_scene=2, base_channels=channels, batch_size=1)
                if resume: cfg['resume'] = str(resume)
                if finetune:
                    cfg['finetune'] = str(finetune)
                    cfg['lr'] = 2e-5
                file = root / (name + '.yaml'); file.write_text(yaml.safe_dump(cfg))
                main(['--stage','combined','--data',str(root/'train'),'--val-data',str(root/'val'),'--config',str(file),'--out',str(root/name)])
                return root / name / 'combined_best.pt'
            # Force CPU even on developer machines with CUDA.
            from unittest.mock import patch
            with patch('torch.cuda.is_available', return_value=False):
                first = train('first',1)
                original = torch.load(first, weights_only=False)
                def inspect_initialization(model, loader, optimizer, device, stage):
                    for key, value in model.state_dict().items():
                        torch.testing.assert_close(value, original['model'][key], rtol=0, atol=0)
                    self.assertEqual(len(optimizer.state), 0)
                    self.assertEqual(optimizer.param_groups[0]['lr'], 2e-5)
                    return 0.0
                with patch('panorama.pano_ai.train.train_restoration.train_one_epoch', side_effect=inspect_initialization):
                    tuned = train('tuned', 1, finetune=first)
                tuned_state = torch.load(tuned, weights_only=False)
                self.assertEqual(tuned_state['epoch'], 1)
                self.assertEqual(tuned_state['finetuned_from'], str(first))
                with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
                    train('both', 2, first, finetune=first)
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    train('bad_tune', 1, channels=16, finetune=first)
                resumed = train('resumed',2,first)
                full = train('full',2)
                a = torch.load(resumed, weights_only=False)
                b = torch.load(full, weights_only=False)
                self.assertEqual(a['epoch'], b['epoch'])
                self.assertEqual(a['selection_loss'], b['selection_loss'])
                for key in a['model']:
                    torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)
                # Recover a latest checkpoint whose epoch was worse than the best.
                def metric(loss):
                    return dict(loss=loss, psnr=1., ssim=0.5)
                with patch('panorama.pano_ai.train.train_restoration.evaluate', side_effect=[metric(.1), metric(.3)]):
                    earlier_best = train('worse', 2)
                latest = earlier_best.with_name('combined_last.pt')
                saved = torch.load(latest, weights_only=False)
                self.assertEqual(saved['epoch'], 2)
                self.assertEqual(saved['best_checkpoint']['epoch'], 1)
                with patch('panorama.pano_ai.train.train_restoration.evaluate', return_value=metric(.4)):
                    recovered = train('recovered', 3, latest)
                self.assertEqual(torch.load(recovered, weights_only=False)['epoch'], 1)
                self.assertEqual(torch.load(recovered.with_name('combined_last.pt'), weights_only=False)['epoch'], 3)
                last_resume = train('last_resume', 2, first.with_name('combined_last.pt'))
                last_weights = torch.load(last_resume, weights_only=False)['model']
                for key in b['model']:
                    torch.testing.assert_close(last_weights[key], b['model'][key], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    train('bad',2,first,16)
                with self.assertRaisesRegex(ValueError, 'epochs must exceed'):
                    train('finished',1,first)

    def test_failed_write_preserves_previous_checkpoint(self):
        from unittest.mock import patch
        from panorama.pano_ai.train.train_restoration import save_checkpoint
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'last.pt'
            save_checkpoint({'epoch': 1}, path)
            with patch('torch.save', side_effect=OSError('disk full')):
                with self.assertRaises(OSError):
                    save_checkpoint({'epoch': 2}, path)
            self.assertEqual(torch.load(path, weights_only=False)['epoch'], 1)
            self.assertEqual(list(Path(tmp).iterdir()), [path])
