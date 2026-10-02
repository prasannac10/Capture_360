"""Train one correction module independently from paired before/after samples."""

import argparse
import copy
import tempfile
import os
import random
from pathlib import Path
import numpy as np

import torch
import yaml
from torch.utils.data import DataLoader, random_split

from ..data.correction_dataset import CorrectionPairDataset
from ..models.color_enhance import ColorEnhancementUNet
from ..models.glare_removal import GlareRemovalUNet
from ..models.nadir_zenith import NadirZenithInpainter
from ..models.advanced_corrections import GhostRemovalUNet
from ..models.combined_restoration import CombinedRestorationUNet
from ..data.paired_panorama import PairedPanoramaDataset
from .restoration import evaluate, train_one_epoch

MODELS = {
    "combined": CombinedRestorationUNet,
    "glare": GlareRemovalUNet,
    "nadir_zenith": NadirZenithInpainter,
    "color": ColorEnhancementUNet,
    "ghost_removal": GhostRemovalUNet,
}


def save_checkpoint(state, path):
    """Replace a checkpoint only after its complete payload has been written."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=sorted(MODELS), required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--val-data", help="Explicit held-out scene directory")
    p.add_argument('--mobile-val-data', help='Separate reviewed real-mobile panorama pairs')
    p.add_argument("--config", required=True)
    p.add_argument("--out", default="checkpoints")
    args = p.parse_args(argv)

    with open(args.config, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if cfg.get('resume') and cfg.get('finetune'):
        raise ValueError('resume and finetune are mutually exclusive')
    def dataset(path):
        size = tuple(cfg.get('input_size', [256, 512]))
        return (PairedPanoramaDataset(path, size, cfg.get('crops_per_scene', 12), random_crops=path == args.data) if args.stage == 'combined'
                else CorrectionPairDataset(path, args.stage, size))
    if args.stage == 'combined' and not args.val_data:
        raise ValueError('Combined training requires --val-data with separate held-out scene IDs')
    ds = dataset(args.data)
    if len(ds) < (1 if args.val_data else 2):
        raise RuntimeError(
            f"Need paired intermediate artifacts for {args.stage}; found {len(ds)} pairs"
        )

    if args.val_data:
        if Path(args.data).resolve() == Path(args.val_data).resolve():
            raise ValueError('Use separate training and validation scene folders')
        train_ds = ds
        val_ds = dataset(args.val_data)
        if args.stage == 'combined' and ds.source_scene_ids & val_ds.source_scene_ids:
            raise ValueError('Training and validation contain the same scene IDs')
        if len(val_ds) == 0:
            raise ValueError('No held-out restoration pairs')
    else:
        val_count = max(1, round(0.1 * len(ds)))
        train_ds, val_ds = random_split(ds, [len(ds) - val_count, val_count], generator=torch.Generator().manual_seed(42))
    bs = cfg.get("batch_size", 4)
    loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    vloader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    mobile_loader = None
    if args.mobile_val_data:
        if args.stage != 'combined':
            raise ValueError('Mobile paired validation currently supports combined restoration only')
        mobile_ds = PairedPanoramaDataset(args.mobile_val_data, tuple(cfg.get('input_size', [256, 512])),
                                          cfg.get('crops_per_scene', 12), required_domain='real_mobile')
        if mobile_ds.source_scene_ids & (ds.source_scene_ids | val_ds.source_scene_ids):
            raise ValueError('Mobile validation must use physical scenes absent from training and paired validation')
        mobile_loader = DataLoader(mobile_ds, batch_size=bs, shuffle=False)
    selection = cfg.get('checkpoint_selection', 'mobile' if mobile_loader is not None else 'paired')
    if selection not in ('paired', 'mobile') or (selection == 'mobile' and mobile_loader is None):
        raise ValueError('checkpoint_selection=mobile requires real mobile validation pairs')
    channel_key = 'base_channels' if args.stage == 'ghost_removal' else 'channels'
    model = MODELS[args.stage](**{channel_key: cfg.get("base_channels", 32)})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    mobile_baseline = evaluate(torch.nn.Identity().to(device), mobile_loader, device, stage='combined') if mobile_loader is not None else None
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    os.makedirs(args.out, exist_ok=True)
    best = float("inf")
    best_state = None
    start_epoch = 0
    finetune = cfg.get('finetune')
    if finetune:
        finetune = (Path(args.config).resolve().parent / finetune).resolve()
        source = torch.load(finetune, map_location=device, weights_only=False)
        contract = 'paired_panorama_restoration_v1' if args.stage == 'combined' else 'native_masked_restoration_v1'
        if (source.get('task') != args.stage or source.get('contract') != contract
                or source.get('channels') != cfg.get('base_channels', 32)):
            raise ValueError('Fine-tuning checkpoint task, contract or channels mismatch')
        if not all(torch.isfinite(value).all() for value in source['model'].values()):
            raise ValueError('Non-finite fine-tuning weights')
        model.load_state_dict(source['model'], strict=True)
    resume = cfg.get('resume')
    if resume:
        resume = (Path(args.config).resolve().parent / resume).resolve()
        state = torch.load(resume, map_location=device, weights_only=False)
        contract = 'paired_panorama_restoration_v1' if args.stage == 'combined' else 'native_masked_restoration_v1'
        if (state.get('task') != args.stage or state.get('contract') != contract
                or state.get('channels') != cfg.get('base_channels', 32)
                or state.get('checkpoint_selection') != selection):
            raise ValueError('Resume checkpoint task, contract, channels or selection mismatch')
        start_epoch = int(state['epoch'])
        if not 0 < start_epoch < cfg.get('epochs', 20):
            raise ValueError('epochs must exceed the completed checkpoint epoch')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        best_state = state.get('best_checkpoint', state)
        best = float(best_state['selection_loss'])
        if not np.isfinite(best):
            raise ValueError('Non-finite resume selection score')
        rng = state.get('rng')
        if rng:
            random.setstate(rng['python'])
            np.random.set_state(rng['numpy'])
            torch.set_rng_state(rng['torch'].cpu())
            if device.type == 'cuda' and rng.get('cuda') is not None:
                torch.cuda.set_rng_state_all([value.cpu() for value in rng['cuda']])
        # Preserve the selected checkpoint even if later epochs do not improve.
        save_checkpoint(best_state, os.path.join(args.out, f"{args.stage}_best.pt"))
    initial_parameter = next(model.parameters()).detach().clone()
    for epoch in range(start_epoch, cfg.get("epochs", 20)):
        loss = train_one_epoch(model, loader, optimizer, device, stage=args.stage)
        metrics = evaluate(model, vloader, device, stage=args.stage)
        mobile_metrics = evaluate(model, mobile_loader, device, stage=args.stage) if mobile_loader is not None else None
        score = (mobile_metrics if selection == 'mobile' else metrics)['loss']
        if not torch.isfinite(torch.tensor(score)):
            raise ValueError('Non-finite validation score')
        print(
            f"epoch {epoch + 1} | train={loss:.6f} | val={metrics['loss']:.6f} | psnr={metrics['psnr']:.3f} | ssim={metrics['ssim']:.4f}"
        )
        if mobile_metrics is not None:
            print(f"mobile_val={mobile_metrics['loss']:.6f} selection={selection}")
        state = {
            "model": model.state_dict(),
            "finetuned_from": str(finetune) if finetune else None,
            "resumed_from": str(resume) if resume else None,
            "rng": {'python': random.getstate(), 'numpy': np.random.get_state(),
                    'torch': torch.get_rng_state(),
                    'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None},
            "parameter_change_l1": float((next(model.parameters()).detach() - initial_parameter).abs().sum()),
            "task": args.stage,
            "contract": "paired_panorama_restoration_v1" if args.stage == 'combined' else "native_masked_restoration_v1",
            "channels": cfg.get("base_channels", 32),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "metrics": metrics,
            "mobile_metrics": mobile_metrics,
            "mobile_baseline": mobile_baseline,
            "mobile_beats_baseline_l1": mobile_metrics['loss'] < mobile_baseline['loss'] if mobile_metrics is not None else None,
            "checkpoint_selection": selection,
            "selection_loss": score,
        }
        if score < best:
            best = score
            best_state = copy.deepcopy(state)
            save_checkpoint(best_state, os.path.join(args.out, f"{args.stage}_best.pt"))
        # Embed the earlier best checkpoint so latest recovery is self-contained.
        state['best_checkpoint'] = best_state
        save_checkpoint(state, os.path.join(args.out, f"{args.stage}_last.pt"))



if __name__ == "__main__":
    main()
