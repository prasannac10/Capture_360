"""Native-resolution tiled panorama training with validation and EMA."""

import argparse
from pathlib import Path
import yaml, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data.tile_dataset import (
    VariableTilePanoramaDataset,
    tile_collate,
    iter_tile_batches,
)
from .models.panorama_model import PanoramaModel
from .models.tile_spherical import panorama_contract
from .data.native_rgb import NativeRGBSource
from .utils.ema import EMA
from .train.panorama_loss import PanoramaLoss
from .train.metrics import image_quality, seam_quality
from .train.logging import TrainingLogger
from panorama.stitching.profiles import validate_frame_set


def run_scene(
    model, sample, device, tile_batch_size, train, optimizer=None, scaler=None, criterion=None, ema=None,
    return_metrics=False,
):
    # tile_size/overlap are supplied by caller through model attributes in the closure below.
    if train:
        optimizer.zero_grad(set_to_none=True)
    native = getattr(model.decoder, 'detail_mode', 'features') == 'rgb_residual'
    source_sampler = NativeRGBSource(sample) if native else None
    output_region = None
    if native:
        oh, ow = model.decoder.output_size
        if tuple(sample['gt_panorama'].shape[-2:]) != (oh, ow):
            raise ValueError('Native RGB detail training requires ground truth at the configured output resolution; do not upscale low-resolution targets')
        crop = int(sample.get('_native_crop_size', 1024))
        if crop <= 0:
            raise ValueError('native_crop_size must be positive')
        ch, cw = min(crop, oh), min(crop, ow)
        if train:
            top = int(torch.randint(oh - ch + 1, (1,)))
            left = int(torch.randint(ow - cw + 1, (1,)))
            # Oversample overlaps with actual view disagreement. Validation
            # uses fixed crops and does not depend on the training sampler.
            ph, pw = min(128, oh), min(256, ow)
            _, coverage, disagreement = source_sampler((0, 0, ph, pw), (ph, pw), device)
            supported = coverage[0, 0] > 0
            candidates = torch.nonzero(supported)
            if float(torch.rand(())) < sample.get('_seam_crop_probability', .5):
                seams = torch.nonzero((source_sampler.last_diagnostics['seam_mask'][0, 0] > 0) & supported)
                if len(seams):
                    candidates = seams
            if not len(candidates):
                raise ValueError('Native crop training requires observed source coverage')
            row, column = candidates[int(torch.randint(len(candidates), (1,)))].tolist()
            top = max(0, min(oh - ch, round((row + .5) * oh / ph) - ch // 2))
            left = max(0, min(ow - cw, round((column + .5) * ow / pw) - cw // 2))
        else:
            top, left = sample.get('_validation_crop_origin', ((oh - ch) // 2, (ow - cw) // 2))
        output_region = (top, left, ch, cw)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx, torch.autocast(device_type=device.type, enabled=bool(scaler and scaler.is_enabled())):
        pred = model.forward_scene(
            lambda: iter_tile_batches(
                sample, sample["_tile_size"], sample["_overlap"], tile_batch_size
            ),
            sample["image_size"],
            sample["camera_params"],
            sample["poses"],
            tile_batch_size,
            source_sampler=source_sampler,
            output_region=output_region,
        )
        gt = sample['gt_panorama']
        if output_region is not None:
            top, left, ch, cw = output_region
            gt = gt[:, top:top + ch, left:left + cw]
        gt = gt.unsqueeze(0).to(device)
        if gt.shape[-2:] != pred.shape[-2:]:
            gt = F.interpolate(gt, pred.shape[-2:], mode='bilinear', align_corners=False, antialias=True)
        coverage = None
        if native:
            _, coverage, _ = source_sampler(output_region, model.decoder.output_size, device)
            if not bool(coverage.any()):
                raise ValueError('Training/evaluation crop has no observed source pixels')
        if criterion is not None:
            loss = criterion(pred, gt, mask=coverage, seam_mask=source_sampler.last_diagnostics['seam_mask']) if native else criterion(pred, gt)
        elif coverage is not None:
            loss = ((pred - gt).abs() * coverage).sum() / (coverage.sum() * pred.shape[1]).clamp_min(1)
        else:
            loss = F.l1_loss(pred, gt)
        if not torch.isfinite(loss):
            raise ValueError('Non-finite panorama training loss')
        quality = image_quality(pred, gt, mask=coverage) if return_metrics else None
        if return_metrics and native:
            quality.update(seam_quality(pred, gt, source_sampler.last_diagnostics['seam_mask']))
        if train:
            if scaler and scaler.is_enabled():
                old_scale = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= old_scale
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                stepped = True
            if ema is not None and stepped:
                ema.update(model)
    value = float(loss.detach().item())
    return dict(quality, loss=value) if return_metrics else value


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="../stitching/config.yaml")
    p.add_argument("--epochs", type=int)
    a = p.parse_args(argv)
    root = Path(__file__).resolve().parent
    cfg = yaml.safe_load((root / a.config).read_text())
    m = cfg["model"]
    tr = cfg["training"]
    if tr.get('mode', 'supervised') != 'supervised':
        raise NotImplementedError('The panorama trainer currently supports supervised mode only')
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = PanoramaLoss(cfg.get('loss', {})).to(dev)
    tile_bs = int(
        tr.get("tile_batch_size", cfg.get("inference", {}).get("tile_batch_size", 4))
    )
    train = VariableTilePanoramaDataset(
        root / tr["training_data"],
        True,
        m["tile_size"],
        m["tile_overlap"],
        cfg["input"]["min_frames"],
        cfg["input"]["max_frames"],
        tr.get("max_tiles_per_frame"),
        input_config=cfg['input'],
    )
    val = (
        VariableTilePanoramaDataset(
            root / tr["val_data"],
            True,
            m["tile_size"],
            m["tile_overlap"],
            cfg["input"]["min_frames"],
            cfg["input"]["max_frames"],
            tr.get("max_tiles_per_frame"),
            input_config=cfg['input'],
        )
        if tr.get("val_data") and (root / tr["val_data"]).exists()
        else None
    )
    model = PanoramaModel(
        m["feature_dim"],
        (m["pano_feature_height"], m["pano_feature_width"]),
        (m["train_output_height"], m["train_output_width"]),
        m["tile_size"],
        m["encoder"]["backbone"],
        m["encoder"]["pretrained"],
        m["attention"]["heads"],
        m["attention"].get("layers", 2),
        tr.get("decoder_output_tile", 1024),
        detail_mode=m.get('detail', {}).get('mode', 'features'),
        residual_scale=m.get('detail', {}).get('residual_scale', .1),
        checkpoint_encoder=tr.get('checkpoint_encoder', False),
    ).to(dev)
    initial_parameter = next(model.parameters()).detach().clone()
    opt = torch.optim.AdamW(
        model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"]
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=bool(tr.get("mixed_precision", True) and dev.type == "cuda")
    )
    ema = EMA(model, tr["ema_decay"]) if tr.get("use_ema", False) else None
    out = root / tr["model_path"]
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    epochs = a.epochs or tr["epochs"]
    native = m.get('detail', {}).get('mode', 'features') == 'rgb_residual'
    crops_per_scene = int(tr.get('native_crops_per_scene', 4)) if native else 1
    if crops_per_scene <= 0:
        raise ValueError('native_crops_per_scene must be positive')
    logger = TrainingLogger(out, 'panorama', model, opt, cfg, device=str(dev), epochs=epochs,
                            train_scenes=len(train), validation_scenes=len(val) if val else 0,
                            mixed_precision=scaler.is_enabled(),
                            validation_weights='ema' if ema else 'model',
                            loss_definition='weighted panorama objective; l1 is logged separately',
                            evaluation_scope='five fixed plus up to three ownership-boundary native crops per scene' if native else 'full panorama')

    def prep(s):
        s = dict(s)
        name, capture_profile = validate_frame_set(
            [tuple(size.int().tolist()) for size in s["image_size"]], cfg["input"]
        )
        if (
            not capture_profile["min_frames"]
            <= len(s["frame_paths"])
            <= capture_profile["max_frames"]
        ):
            raise ValueError(
                f"{s['scene']} ({name}) requires {capture_profile['min_frames']}..{capture_profile['max_frames']} frames"
            )
        expected_projection = (
            0.0 if capture_profile["projection"] == "fisheye_180" else 1.0
        )
        if not bool(torch.all(s["camera_params"][:, 4] == expected_projection)):
            raise ValueError(
                f"{s['scene']} ({name}) camera.json projection does not match its configured profile"
            )
        s["_tile_size"] = m["tile_size"]
        s["_overlap"] = m["tile_overlap"]
        s['_native_crop_size'] = tr.get('native_crop_size', 1024)
        s['_seam_crop_probability'] = tr.get('seam_crop_probability', .5)
        if not 0 <= s['_seam_crop_probability'] <= 1:
            raise ValueError('seam_crop_probability must be in [0, 1]')
        return s

    for ep in range(1, epochs + 1):
        model.train()
        train_rows = [
            run_scene(model, prep(train[i]), dev, tile_bs, True, opt, scaler, criterion, ema, return_metrics=True)
            for i in torch.randperm(len(train)).tolist()
            for _ in range(crops_per_scene)
        ]
        training_metrics = {key: sum(row[key] for row in train_rows) / len(train_rows) for key in train_rows[0]}
        tl = training_metrics['loss']
        if ema:
            ema.copy_to(model)
        model.eval()
        validation_metrics = None
        if val:
            val_rows = []
            for i in range(len(val)):
                sample = prep(val[i])
                if native:
                    oh, ow = model.decoder.output_size
                    ch, cw = min(sample['_native_crop_size'], oh), min(sample['_native_crop_size'], ow)
                    origins = list(dict.fromkeys([(0, 0), (0, ow - cw), (oh - ch, 0),
                                                   (oh - ch, ow - cw), ((oh - ch) // 2, (ow - cw) // 2)]))
                    preview = NativeRGBSource(sample)
                    ph, pw = min(128, oh), min(256, ow)
                    preview((0, 0, ph, pw), (ph, pw), dev)
                    boundary_pixels = torch.nonzero(preview.last_diagnostics['seam_mask'][0, 0] > 0)
                    if len(boundary_pixels):
                        for index in sorted(set((len(boundary_pixels)//4, len(boundary_pixels)//2, 3*len(boundary_pixels)//4))):
                            row, col = boundary_pixels[index].tolist()
                            origin = (max(0,min(oh-ch,round((row+.5)*oh/ph)-ch//2)),
                                      max(0,min(ow-cw,round((col+.5)*ow/pw)-cw//2)))
                            if origin not in origins:
                                origins.append(origin)
                else:
                    origins = [None]
                for origin in origins:
                    if origin is not None:
                        sample['_validation_crop_origin'] = origin
                        _, coverage, _ = NativeRGBSource(sample)((*origin, ch, cw), (oh, ow), dev)
                        if not bool(coverage.any()):
                            continue
                    val_rows.append(run_scene(model, sample, dev, tile_bs, False, criterion=criterion,
                                              return_metrics=True))
            if not val_rows:
                raise ValueError('Validation scenes have no observed source coverage')
            validation_metrics = {key: sum(row[key] for row in val_rows) / len(val_rows) for key in val_rows[0]}
        vl = validation_metrics['loss'] if validation_metrics else tl
        if ema:
            ema.restore(model)
        state = {
            "parameter_change_l1": float((next(model.parameters()).detach() - initial_parameter).abs().sum()),
            "contract": panorama_contract(m.get('detail', {}).get('mode', 'features')),
            "task": "panorama",
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": ep,
            "train_loss": tl,
            "val_loss": vl,
            "train_metrics": training_metrics,
            "validation_metrics": validation_metrics,
            "learning_rates": logger.learning_rates(opt),
            "ema": ema.shadow if ema else None,
            "config": cfg,
        }
        torch.save(state, out / tr["model_name"])
        if vl < best:
            best = vl
            torch.save(state, out / tr["best_model_name"])
        logger.epoch(ep, opt, training_metrics, validation_metrics)


if __name__ == "__main__":
    main()
