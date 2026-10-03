import math
import torch


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2).item()
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def ssim(pred, target):
    # Lightweight global SSIM proxy; use torchmetrics/skimage for full SSIM in production.
    mu_x, mu_y = pred.mean(), target.mean()
    vx, vy = pred.var(correction=0), target.var(correction=0)
    c1 = 0.01**2
    c2 = 0.03**2
    cov = ((pred - mu_x) * (target - mu_y)).mean()
    return float(
        ((2 * mu_x * mu_y + c1) * (2 * cov + c2))
        / ((mu_x**2 + mu_y**2 + c1) * (vx + vy + c2))
    )


@torch.no_grad()
def image_quality(pred, target, mask=None):
    """Mean per-image metrics; pixel accuracy requires ALL RGB channels within 8/255."""
    pred, target = pred.detach().float(), target.detach().float()
    rows = []
    for index, (prediction, reference) in enumerate(zip(pred, target)):
        if mask is not None:
            valid = mask[index, 0] > 0
            if not bool(valid.any()):
                continue
            prediction, reference = prediction[:, valid], reference[:, valid]
        rows.append(dict(psnr=psnr(prediction, reference), ssim=ssim(prediction, reference),
                         pixel_accuracy_pct=float(((prediction-reference).abs().amax(0) <= 8/255).float().mean() * 100),
                         l1=float((prediction-reference).abs().mean())))
    return {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]} if rows else {}


def lpips(pred, target):
    try:
        import lpips

        net = lpips.LPIPS(net="alex").to(pred.device)
        return float(net(pred * 2 - 1, target * 2 - 1).mean().detach().cpu())
    except ImportError:
        return None
