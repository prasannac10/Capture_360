"""Paired structural/detail objectives without pretrained downloads or GANs."""
import math
import torch
from torch.nn import functional as F


DEFAULT_LOSS = dict(l1=1.0, ssim=.2, gradient=.2, multiscale=.1, changed_regions=2.0)


def loss_settings(settings=None):
    weights = dict(DEFAULT_LOSS)
    if settings:
        if set(settings) - weights.keys():
            raise ValueError('Unknown combined restoration loss weight')
        weights.update(settings)
    if any(not math.isfinite(v) or v < 0 for v in weights.values()) or weights['l1'] <= 0:
        raise ValueError('Restoration loss weights must be nonnegative with positive L1')
    return weights


def combined_loss(prediction, target, original, settings=None):
    weights = loss_settings(settings)
    prediction, target, original = prediction.float(), target.float(), original.float()
    # Retain supervision on unchanged pixels, and emphasize edited defect regions.
    changed = ((target - original).abs().mean(1, keepdim=True) / .1).clamp(0, 1).detach()
    importance = 1 + weights['changed_regions'] * changed
    loss = weights['l1'] * ((prediction - target).abs() * importance).sum() / (
        importance.sum() * prediction.shape[1]).clamp_min(1)
    if weights['gradient']:
        for dim in (-1, -2):
            if prediction.shape[dim] > 1:
                loss = loss + weights['gradient'] * .5 * F.l1_loss(
                    prediction.diff(dim=dim), target.diff(dim=dim))
    if weights['multiscale']:
        for scale in (2, 4):
            if min(prediction.shape[-2:]) >= scale:
                loss = loss + weights['multiscale'] * .5 * F.l1_loss(
                    F.avg_pool2d(prediction, scale), F.avg_pool2d(target, scale))
    if weights['ssim']:
        window = min(7, *prediction.shape[-2:])
        def mean(x):
            return F.avg_pool2d(x, window, stride=1)
        a, b = mean(prediction), mean(target)
        va = (mean(prediction.square()) - a.square()).clamp_min(0)
        vb = (mean(target.square()) - b.square()).clamp_min(0)
        covariance = mean(prediction * target) - a * b
        similarity = ((2 * a * b + .01 ** 2) * (2 * covariance + .03 ** 2) /
                      ((a.square() + b.square() + .01 ** 2) * (va + vb + .03 ** 2)))
        loss = loss + weights['ssim'] * (1 - similarity).clamp_min(0).mean()
    return loss
