"""Losses actually supported by the RGB panorama trainer."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class PanoramaLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        if any(float(value) != 0 for value in config.get('geometry', {}).values()):
            raise NotImplementedError('Geometry loss needs geometry predictions; set geometry weights to zero')
        self.weights = {'l1_weight': 1., 'ssim_weight': 0., 'perceptual_weight': 0., 'edge_weight': 0.}
        supplied = config.get('supervised', {})
        if set(supplied) - self.weights.keys():
            raise ValueError('Unknown supervised loss weight')
        self.weights.update(supplied)
        if any(not math.isfinite(float(v)) or float(v) < 0 for v in self.weights.values()) or not any(self.weights.values()):
            raise ValueError('Loss weights must be finite, nonnegative, and not all zero')
        self.perceptual = None
        if self.weights['perceptual_weight']:
            from torchvision.models import resnet18, ResNet18_Weights
            net = resnet18(weights=ResNet18_Weights.DEFAULT)
            self.perceptual = nn.Sequential(*list(net.children())[:6]).eval().requires_grad_(False)
        self.register_buffer('mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    def forward(self, prediction, target, mask=None):
        prediction, target = prediction.float(), target.float()
        if mask is not None:
            mask = mask.to(prediction).expand_as(prediction)
        def reduce(error, valid=mask):
            return error.mean() if valid is None else (error * valid).sum() / valid.sum().clamp_min(1)
        loss = self.weights['l1_weight'] * reduce((prediction - target).abs())
        if self.weights['edge_weight']:
            edges = []
            if prediction.shape[-1] > 1:
                edges.append(reduce(((prediction[..., 1:] - prediction[..., :-1]) -
                                     (target[..., 1:] - target[..., :-1])).abs(),
                                    None if mask is None else mask[..., 1:] * mask[..., :-1]))
            if prediction.shape[-2] > 1:
                edges.append(reduce(((prediction[..., 1:, :] - prediction[..., :-1, :]) -
                                     (target[..., 1:, :] - target[..., :-1, :])).abs(),
                                    None if mask is None else mask[..., 1:, :] * mask[..., :-1, :]))
            if edges:
                loss = loss + self.weights['edge_weight'] * torch.stack(edges).mean()
        if self.weights['ssim_weight']:
            size = min(7, *prediction.shape[-2:])
            def mean(x):
                return F.avg_pool2d(x, size, stride=1)
            a, b = mean(prediction), mean(target)
            va = (mean(prediction.square()) - a.square()).clamp_min(0)
            vb = (mean(target.square()) - b.square()).clamp_min(0)
            covariance = mean(prediction * target) - a * b
            similarity = ((2 * a * b + .01 ** 2) * (2 * covariance + .03 ** 2)
                          / ((a.square() + b.square() + .01 ** 2) * (va + vb + .03 ** 2)))
            loss = loss + self.weights['ssim_weight'] * reduce(1 - similarity, None if mask is None else (mean(mask) >= 1 - 1e-6).to(mask))
        if self.perceptual is not None:
            self.perceptual.eval()
            # Bound perceptual feature memory; full-resolution L1 still applies.
            size = (min(512, prediction.shape[-2]), min(1024, prediction.shape[-1]))
            # Identical zero context prevents unsupported target edits influencing features.
            if mask is not None:
                prediction, target = prediction * mask, target * mask
            a = F.interpolate(prediction, size, mode='bilinear', align_corners=False)
            b = F.interpolate(target, size, mode='bilinear', align_corners=False)
            with torch.no_grad():
                target_features = self.perceptual((b - self.mean) / self.std)
            loss = loss + self.weights['perceptual_weight'] * F.l1_loss(
                self.perceptual((a - self.mean) / self.std), target_features)
        return loss
