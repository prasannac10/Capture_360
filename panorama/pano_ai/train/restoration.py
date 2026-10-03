import torch
import torch.nn.functional as F
import numpy as np
from panorama.stitching.visual_regression import metrics as visual_metrics

from .metrics import image_quality
from .restoration_loss import combined_loss


def _predict(model, batch, device, stage):
    x = batch["input"].to(device)
    if stage in ('color', 'combined'):
        return model(x)
    if 'mask' not in batch:
        raise ValueError(f'{stage} training requires an explicit defect mask')
    mask = batch['mask'].to(device)
    predicted = model(x, mask) if getattr(model, 'mask_channels', 0) else model(x)
    return torch.where(mask > 0, x + mask * (predicted - x), x)


def restoration_loss(pred, target, batch, device, stage, settings=None):
    error = (pred - target).abs()
    if stage == 'combined':
        return combined_loss(pred, target, batch['input'].to(device), settings)
    if stage == 'color':
        return error.mean()
    mask = batch['mask'].to(device)
    return (error * mask).sum() / (mask.sum() * pred.shape[1]).clamp_min(1)


def train_one_epoch(model, loader, optimizer, device, stage="glare"):
    model.train()
    total = 0.0
    quality_total, samples = {}, 0
    for batch in loader:
        y = batch["target"].to(device)
        pred = _predict(model, batch, device, stage)
        quality = image_quality(pred, y)
        for key, value in quality.items():
            quality_total[key] = quality_total.get(key, 0.) + value * len(y)
        samples += len(y)
        loss = restoration_loss(pred, y, batch, device, stage, getattr(model, 'loss_settings', None))
        if not torch.isfinite(loss):
            raise ValueError('Non-finite restoration training loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        total += loss.item() * len(y)
    model.last_train_metrics = {key: value / max(1, samples) for key, value in quality_total.items()}
    model.last_train_metrics['loss'] = total / max(1, samples)
    return model.last_train_metrics['loss']


def evaluate(model, loader, device, stage="glare"):
    model.eval()
    losses, quality_total, visual, samples = [], {}, [], 0
    with torch.no_grad():
        for batch in loader:
            y = batch["target"].to(device)
            pred = _predict(model, batch, device, stage)
            # Report/select with plain held-out L1, comparable to the unchanged input.
            losses.append(((pred - y).abs().mean() if stage == 'combined' else
                           restoration_loss(pred, y, batch, device, stage)).item() * len(y))
            for key, value in image_quality(pred, y).items():
                quality_total[key] = quality_total.get(key, 0.) + value * len(y)
            samples += len(y)
            for index in range(len(pred)):
                def pixels(t):
                    return np.rint(t[index].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                original = pixels(batch['input']) if stage not in ('color', 'combined') else None
                mask = batch['mask'][index, 0].cpu().numpy() if stage not in ('color', 'combined') else None
                prediction_pixels, target_pixels = pixels(pred), pixels(y)
                row = visual_metrics(prediction_pixels, target_pixels, original, mask)
                if 'longitude_join_column' in batch:
                    # A wrapped native crop contains the real join internally;
                    # its outer edges are not the panorama longitude boundary.
                    row.pop('longitude_join_error')
                    column = int(batch['longitude_join_column'][index])
                    if 0 < column < prediction_pixels.shape[1]:
                        residual = (prediction_pixels.astype(np.float32) - target_pixels.astype(np.float32)) / 255
                        row['longitude_join_error'] = float(np.abs(residual[:, column] - residual[:, column-1]).mean())
                visual.append(row)
    return {
        "loss": sum(losses) / max(1, samples),
        **{key: value / max(1, samples) for key, value in quality_total.items()},
        "visual": {key: sum(row[key] for row in visual if key in row) / sum(key in row for row in visual)
                   for key in set().union(*(row.keys() for row in visual))} if visual else {},
    }


def evaluate_identity(loader, device):
    return evaluate(torch.nn.Identity().to(device), loader, device, stage='combined')
