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
        self.weights = {'l1_weight': 1., 'ssim_weight': 0., 'perceptual_weight': 0.}
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

    def forward(self, prediction, target):
        prediction, target = prediction.float(), target.float()
        loss = self.weights['l1_weight'] * F.l1_loss(prediction, target)
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
            loss = loss + self.weights['ssim_weight'] * (1 - similarity.mean())
        if self.perceptual is not None:
            self.perceptual.eval()
            # Bound perceptual feature memory; full-resolution L1 still applies.
            size = (min(512, prediction.shape[-2]), min(1024, prediction.shape[-1]))
            a = F.interpolate(prediction, size, mode='bilinear', align_corners=False)
            b = F.interpolate(target, size, mode='bilinear', align_corners=False)
            with torch.no_grad():
                target_features = self.perceptual((b - self.mean) / self.std)
            loss = loss + self.weights['perceptual_weight'] * F.l1_loss(
                self.perceptual((a - self.mean) / self.std), target_features)
        return loss
