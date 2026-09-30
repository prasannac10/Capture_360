"""Learned proposals with explicit confidence checks and geometric fallback."""
from pathlib import Path
import numpy as np
import cv2
import torch

from .models.pairwise import PairwiseHead
from panorama.stitching.warp_guard import guard_warp


class GuardedViewRefiner:
    def __init__(self, settings, config_path, device='cpu'):
        self.settings, self.device = settings, torch.device(device)
        self.models = {}
        self.checkpoints = {}
        for task in ('alignment', 'blending'):
            options = settings.get(task, {})
            if not options.get('enabled', False):
                continue
            path = Path(config_path).parent / options['checkpoint']
            state = torch.load(path, map_location=self.device, weights_only=False)
            if state.get('task') != task or state.get('contract') != 'geometric_pairs_v1':
                raise ValueError(f'{path}: wrong pairwise task/coordinate contract')
            if state.get('work_width', 1024) != 1024:
                raise ValueError('Pairwise checkpoint must use a 1024-wide spherical canvas')
            model = PairwiseHead(task, state.get('channels', 32), state.get('max_displacement', 8.)).to(self.device)
            weights = state.get('ema') if options.get('use_ema', True) else None
            model.load_state_dict(weights if weights is not None else state['model'], strict=True)
            self.checkpoints[task] = {'path': str(path.resolve()), 'weights': 'ema' if weights is not None else 'model'}
            self.models[task] = model.eval()

    def predict(self, task, reference, source, reference_valid, source_valid):
        def rgb(a):
            return torch.from_numpy(a[..., ::-1].copy()).permute(2, 0, 1)[None].float().to(self.device) / 255
        def mask(a):
            return torch.from_numpy((a > 0).astype(np.float32))[None, None].to(self.device)
        with torch.inference_mode():
            result = self.models[task](rgb(reference), rgb(source), mask(reference_valid), mask(source_valid))
        return {k: v[0].permute(1, 2, 0).cpu().numpy() for k, v in result.items()}

    def align(self, images, masks, qualities):
        # All proposals are estimated at the training contract's 1024-wide canvas.
        w = 1024
        size = (w, w // 2)
        images = [cv2.resize(im, size, interpolation=cv2.INTER_AREA) for im in images]
        masks = [cv2.resize(m, size, interpolation=cv2.INTER_NEAREST) for m in masks]
        quality = np.array([cv2.resize(q, size) for q in qualities])
        owner = quality.argmax(axis=0)
        reference = np.zeros_like(images[0])
        valid = np.zeros_like(masks[0])
        for i, (im, m) in enumerate(zip(images, masks)):
            take = (owner == i) & (m > 0)
            reference[take], valid[take] = im[take], 255
        fields, reports = [], []
        for i, (im, m) in enumerate(zip(images, masks)):
            zero = np.zeros((*m.shape, 2), np.float32)
            if 'alignment' not in self.models:
                fields.append(zero)
                reports.append({'accepted': False, 'reason': 'disabled', 'fallback': 'geometric_projection'})
                continue
            forward = self.predict('alignment', reference, im, valid, m)
            reverse = self.predict('alignment', im, reference, m, valid)
            field, report = guard_warp(reference, im, valid, m, forward['flow'], forward['confidence'][..., 0], reverse['flow'],
                                       min_confidence=self.settings.get('min_confidence', .8),
                                       max_displacement=self.settings.get('max_displacement', 8.))
            fields.append(field)
            reports.append(report)
        return fields, {'method': 'guarded_learned_alignment', 'frames': reports, 'work_size': list(size)}

    def ownership(self, images, masks, qualities, geometric):
        if 'blending' not in self.models:
            return geometric, {'method': 'geometric_seams', 'reason': 'learned_blending_disabled'}
        # Compare each valid view against its geometric seam reference. Only
        # high-confidence proposals may replace geometric ownership.
        reference = np.zeros_like(images[0])
        owner = np.full(masks[0].shape, -1, np.int16)
        for i, (im, mask) in enumerate(zip(images, geometric)):
            reference[mask > 0], owner[mask > 0] = im[mask > 0], i
        valid = (owner >= 0).astype(np.uint8) * 255
        best = np.full(masks[0].shape, .5, np.float32)
        changed = np.zeros_like(valid, dtype=bool)
        for i, (im, mask) in enumerate(zip(images, masks)):
            size = (1024, 512)
            prediction = self.predict('blending', cv2.resize(reference, size, interpolation=cv2.INTER_AREA),
                                      cv2.resize(im, size, interpolation=cv2.INTER_AREA),
                                      cv2.resize(valid, size, interpolation=cv2.INTER_NEAREST),
                                      cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST))
            prediction = {k: cv2.resize(v, (mask.shape[1], mask.shape[0]))[..., None]
                          for k, v in prediction.items()}
            score = 1 - prediction['weight'][..., 0]
            use = ((mask > 0) & (valid > 0) & (score > best)
                   & (prediction['confidence'][..., 0] >= self.settings.get('min_confidence', .8)))
            owner[use], best[use], changed[use] = i, score[use], True
        return [(owner == i).astype(np.uint8) * 255 for i in range(len(images))], {
            'method': 'confidence_gated_source_selection', 'changed_pixels': int(changed.sum()),
            'fallback_pixels': int(((valid > 0) & ~changed).sum())}
