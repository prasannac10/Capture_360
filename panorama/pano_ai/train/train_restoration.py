"""Train one correction module independently from paired before/after samples."""

import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from ..data.correction_dataset import CorrectionPairDataset
from ..models.color_enhance import ColorEnhancementUNet
from ..models.glare_removal import GlareRemovalUNet
from ..models.nadir_zenith import NadirZenithInpainter
from .restoration import evaluate, train_one_epoch

MODELS = {
    "glare": GlareRemovalUNet,
    "nadir_zenith": NadirZenithInpainter,
    "color": ColorEnhancementUNet,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=sorted(MODELS), required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    ds = CorrectionPairDataset(
        args.data, args.stage, tuple(cfg.get("input_size", [256, 512]))
    )
    if len(ds) < 2:
        raise RuntimeError(
            f"Need paired intermediate artifacts for {args.stage}; found {len(ds)} pairs"
        )

    val_count = max(1, round(0.1 * len(ds)))
    train_ds, val_ds = random_split(
        ds,
        [len(ds) - val_count, val_count],
        generator=torch.Generator().manual_seed(42),
    )
    bs = cfg.get("batch_size", 4)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    vloader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    model = MODELS[args.stage](channels=cfg.get("base_channels", 32))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    os.makedirs(args.out, exist_ok=True)
    best = float("inf")
    for epoch in range(cfg.get("epochs", 20)):
        loss = train_one_epoch(model, loader, optimizer, device, stage=args.stage)
        metrics = evaluate(model, vloader, device, stage=args.stage)
        print(
            f"epoch {epoch + 1} | train={loss:.6f} | val={metrics['loss']:.6f} | psnr={metrics['psnr']:.3f} | ssim={metrics['ssim']:.4f}"
        )
        if metrics["loss"] < best:
            best = metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch + 1,
                    "metrics": metrics,
                },
                os.path.join(args.out, f"{args.stage}_best.pt"),
            )


if __name__ == "__main__":
    main()
