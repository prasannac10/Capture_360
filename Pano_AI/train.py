import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.collate import panorama_collate_fn
from data.dataset import PanoramaDataset
from losses.geometry import geometry_loss
from losses.supervised import supervised_loss
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.encoder import ImageEncoder
from models.panorama_model import PanoramaModel
from utils.checkpoint import save_checkpoint
from utils.ema import EMA


def build_model(cfg, pretrained=None):
    enc_cfg = cfg["model"].get("encoder", {})
    use_pretrained = enc_cfg.get("pretrained", True) if pretrained is None else pretrained
    encoder = ImageEncoder(
        cfg["model"]["feature_dim"],
        backbone=enc_cfg.get("backbone", "resnet18"),
        pretrained=use_pretrained,
    )
    return PanoramaModel(
        encoder,
        SetAggregator(cfg["model"]["feature_dim"]),
        PanoramaDecoder(cfg["model"]["feature_dim"]),
        cfg["model"]["pano_height"],
        cfg["model"]["pano_width"],
    )


def _loss(cfg, mode, pred, batch, device):
    if mode == "supervised":
        return supervised_loss(
            pred,
            batch["gt_panorama"].to(device),
            cfg["loss"]["supervised"].get("l1_weight", 1.0),
            cfg["loss"]["supervised"].get("ssim_weight", 0.0),
        )
    return geometry_loss(
        pred,
        batch["images"].to(device),
        batch["rotations"].to(device),
        batch["mask"].to(device),
        cfg["loss"]["geometry"].get("smoothness_weight", 1.0),
    )


def _run_epoch(model, loader, cfg, mode, device, optimizer=None, ema=None):
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for batch in loader:
            images = batch["images"].to(device)
            rotations = batch["rotations"].to(device)
            mask = batch["mask"].to(device)
            camera = batch["camera_params"].to(device)
            pred = model(images, rotations, mask, camera)
            loss = _loss(cfg, mode, pred, batch, device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if ema:
                    ema.update(model)
            total += float(loss.item())
    return total / max(1, len(loader))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()
    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    mode = cfg["training"]["mode"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = PanoramaDataset(cfg["training"]["training_data"], has_gt=mode == "supervised")
    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, collate_fn=panorama_collate_fn, num_workers=0)

    val_loader = None
    val_root = cfg["training"].get("val_data")
    if val_root and os.path.isdir(val_root):
        val_ds = PanoramaDataset(val_root, has_gt=mode == "supervised")
        if len(val_ds):
            val_loader = DataLoader(val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, collate_fn=panorama_collate_fn, num_workers=0)

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"].get("weight_decay", 1e-4))
    ema = EMA(model, cfg["training"]["ema_decay"]) if cfg["training"].get("use_ema", False) else None
    os.makedirs(cfg["training"]["model_path"], exist_ok=True)

    best = float("inf")
    best_name = cfg["training"].get("best_model_name", "panorama_best.pt")
    last_name = cfg["training"].get("model_name", "panorama_last.pt")
    for epoch in range(cfg["training"]["epochs"]):
        train_loss = _run_epoch(model, train_loader, cfg, mode, device, optimizer, ema)
        val_loss = train_loss
        if val_loader:
            if ema:
                ema.copy_to(model)
            val_loss = _run_epoch(model, val_loader, cfg, mode, device)
            if ema:
                ema.restore(model)
        print(f"epoch {epoch + 1}/{cfg['training']['epochs']} | train={train_loss:.6f} | val={val_loss:.6f}")
        if val_loss < best:
            best = val_loss
            save_checkpoint(os.path.join(cfg["training"]["model_path"], best_name), model, optimizer=optimizer, epoch=epoch + 1, ema=ema)

    save_checkpoint(os.path.join(cfg["training"]["model_path"], last_name), model, optimizer=optimizer, epoch=cfg["training"]["epochs"], ema=ema)
    print("saved:", best_name, "and", last_name)


if __name__ == "__main__":
    main()
