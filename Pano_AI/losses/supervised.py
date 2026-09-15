import torch.nn.functional as F


def _ssim(x, y, window=7):
    mu_x = F.avg_pool2d(x, window, 1, window // 2)
    mu_y = F.avg_pool2d(y, window, 1, window // 2)
    sigma_x = F.avg_pool2d(x * x, window, 1, window // 2) - mu_x * mu_x
    sigma_y = F.avg_pool2d(y * y, window, 1, window // 2) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(x * y, window, 1, window // 2) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2) / ((mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2))).mean()


def supervised_loss(pred, gt, l1_weight=1.0, ssim_weight=0.0):
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction/target shape mismatch: {pred.shape} vs {gt.shape}")
    l1 = F.l1_loss(pred, gt)
    return l1_weight * l1 if ssim_weight <= 0 else l1_weight * l1 + ssim_weight * (1.0 - _ssim(pred, gt))
