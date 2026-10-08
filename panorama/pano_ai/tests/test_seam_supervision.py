import tempfile
from pathlib import Path
import unittest
import json
from unittest.mock import patch
import torch
from panorama.pano_ai.data.native_rgb import NativeRGBSource
from panorama.pano_ai.train.panorama_loss import PanoramaLoss
from panorama.pano_ai.train.metrics import seam_quality
from panorama.pano_ai.tests import test_native_detail


class SeamSupervisionTests(unittest.TestCase):
    def test_owner_boundary_is_observed_and_photometric_baseline_is_comparable(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = test_native_detail.NativeDetailTests().sample(Path(tmp), [(100,100,100),(160,160,160)])
            sample['poses'] = torch.tensor([[-20.,0.,0.],[20.,0.,0.]])
            source = NativeRGBSource(sample, photometric_blending=False)
            hard, coverage, _ = source((0,0,32,64),(32,64),torch.device('cpu'))
            seam = source.last_diagnostics['seam_mask']
            self.assertGreater(float(seam.sum()),0)
            self.assertEqual(float((seam*(1-coverage)).sum()),0)
            blended = NativeRGBSource(sample)((0,0,32,64),(32,64),torch.device('cpu'))[0]
            self.assertGreater(float((hard-blended).abs().sum()),0)

    def test_seam_loss_focuses_reference_error_and_has_gradients(self):
        prediction = torch.zeros(1,3,8,8,requires_grad=True)
        target = torch.zeros_like(prediction)
        target[...,3:5] = .5
        seam = torch.zeros(1,1,8,8);seam[...,3:5] = 1
        criterion = PanoramaLoss({'supervised':{'l1_weight':0,'seam_weight':1}})
        loss = criterion(prediction,target,torch.ones_like(seam),seam)
        self.assertGreater(float(loss.detach()),0)
        loss.backward()
        self.assertGreater(float(prediction.grad.abs().sum()),0)
        metrics = seam_quality(prediction,target,seam)
        self.assertAlmostEqual(metrics['seam_l1'],.5)
        self.assertEqual(metrics['seam_pixels'],16)
        self.assertEqual(seam_quality(prediction,target,torch.zeros_like(seam))['seam_l1'],0)

    def test_diagnostic_outputs_and_ai_pair_provenance(self):
        from panorama.pano_ai.train.diagnose_seams import diagnose
        from panorama.pano_ai.data.prepare_pairs import prepare
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            scene = root / 'scene';scene.mkdir()
            sample = test_native_detail.NativeDetailTests().sample(scene, [(100,100,100),(160,160,160)])
            sample['poses'] = torch.tensor([[-20.,0.,0.],[20.,0.,0.]])
            source = NativeRGBSource(sample)
            sample['gt_panorama'] = source((0,0,32,64),(32,64),torch.device('cpu'))[0][0]
            config = root / 'config.yaml'
            config.write_text(yaml.safe_dump(dict(model=dict(output_height=32,output_width=64), input={})))
            class Dataset:
                scenes = [scene]
                def __getitem__(self, index):
                    return sample
            with patch('panorama.pano_ai.train.diagnose_seams.VariableTilePanoramaDataset',return_value=Dataset()):
                report = diagnose(scene,config,root/'diagnostic',(0,0,32,64))
            self.assertAlmostEqual(report['metrics']['photometric']['l1'],0)
            self.assertTrue((root/'diagnostic/seam.png').exists())
            self.assertTrue((root/'diagnostic/report.json').exists())
            manifest=root/'pairs.json'
            manifest.write_text(json.dumps(dict(version=1,pairs=[dict(scene_id='example',before='Stitched.jpg')])))
            with self.assertRaisesRegex(ValueError,'provenance'):
                prepare(manifest,root/'bundle',crop_size=8,require_ai_outputs=True)
