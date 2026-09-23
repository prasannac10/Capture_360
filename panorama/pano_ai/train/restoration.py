import torch
import torch.nn.functional as F

from .metrics import psnr, ssim


def _predict(model, batch, device, stage):
    x = batch["input"].to(device)
    if stage == "nadir_zenith":
        # A correction mask can be supplied by a prepared dataset later. For
        # paired baseline data, default to repairing the full crop.
        mask = batch.get("mask")
        if mask is None:
            mask = torch.ones(x.shape[0], 1, x.shape[2], x.shape[3], device=device)
        else:
            mask = mask.to(device)
        return model(x, mask)
    return model(x)


def train_one_epoch(model, loader, optimizer, device, stage="glare"):
    model.train()
    total = 0.0
    for batch in loader:
        y = batch["target"].to(device)
        pred = _predict(model, batch, device, stage)
        loss = F.l1_loss(pred, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))


def evaluate(model, loader, device, stage="glare"):
    model.eval()
    losses, ps, ss = [], [], []
    with torch.no_grad():
        for batch in loader:
            y = batch["target"].to(device)
            pred = _predict(model, batch, device, stage)
            losses.append(F.l1_loss(pred, y).item())
            ps.append(psnr(pred, y))
            ss.append(ssim(pred, y))
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "psnr": sum(ps) / max(1, len(ps)),
        "ssim": sum(ss) / max(1, len(ss)),
    }
