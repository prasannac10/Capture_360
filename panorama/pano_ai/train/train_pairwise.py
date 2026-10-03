"""Train one geometric-view task; never infer supervision from an RGB loss alone."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ..models.pairwise import PairwiseHead, inverse_warp


class GeometricPairs(Dataset):
    """NPZ full spherical canvases, RGB float [0,1], inverse flow in canvas pixels."""
    def __init__(self, root, task):
        self.files = sorted(Path(root).glob('*.npz'))
        self.task = task
        if not self.files:
            raise ValueError(f'No geometric pair samples: {root}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        channels = dict(reference=3, source=3, reference_valid=1, source_valid=1,
                        confidence=1, supervision_valid=1)
        channels['flow' if self.task == 'alignment' else 'weight'] = 2 if self.task == 'alignment' else 1
        result = {}
        with np.load(self.files[index], allow_pickle=False) as data:
            if str(data['contract'].item()) != 'geometric_pairs_v1':
                raise ValueError('Expected geometric_pairs_v1 training coordinates')
            for name, count in channels.items():
                a = np.asarray(data[name], dtype=np.float32)
                if a.ndim == 2:
                    a = a[..., None]
                if a.shape != (512, 1024, count) or not np.isfinite(a).all():
                    raise ValueError(f'{name}: expected finite 512x1024x{count} canvas')
                if name != 'flow' and (a.min() < 0 or a.max() > 1):
                    raise ValueError(f'{name} must be in [0,1]')
                result[name] = torch.from_numpy(a.copy()).permute(2, 0, 1)
        if result['supervision_valid'].sum() == 0:
            raise ValueError('Pair has no supervised pixels')
        return result


def task_loss(model, batch):
    prediction = model(*(batch[k] for k in ('reference', 'source', 'reference_valid', 'source_valid')))
    valid = batch['supervision_valid']
    def mean(error):
        return (error * valid).sum() / (valid.sum() * error.shape[1]).clamp_min(1)
    confidence = mean(F.binary_cross_entropy(prediction['confidence'], batch['confidence'], reduction='none'))
    if model.task == 'alignment':
        if batch['flow'].norm(dim=1).max() > model.max_displacement:
            raise ValueError('Alignment label exceeds checkpoint displacement bound')
        reliable = batch['confidence']
        flow_loss = mean(F.smooth_l1_loss(prediction['flow'], batch['flow'], reduction='none') * reliable)
        # Shared inverse sampler also receives gradients; target uses the identical grid.
        target = inverse_warp(batch['source'], batch['flow'])
        warp_loss = mean((inverse_warp(batch['source'], prediction['flow']) - target).abs() * reliable)
        return flow_loss + warp_loss + confidence
    # Inference chooses a source owner, so supervise ownership directly, not an
    # averaged RGB target that encourages ghosting.
    return mean(F.binary_cross_entropy(prediction['weight'], batch['weight'], reduction='none')) + confidence


def main(argv=None):
    from .logging import TrainingLogger
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('alignment', 'blending'), required=True)
    parser.add_argument('--train', required=True)
    parser.add_argument('--val', required=True, help='Held-out scenes, separate from training')
    parser.add_argument('--out', required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args(argv)
    if Path(args.train).resolve() == Path(args.val).resolve():
        raise ValueError('Training and validation must be distinct scene splits')
    if args.epochs <= 0 or args.channels <= 0 or args.lr <= 0:
        raise ValueError('epochs, channels and lr must be positive')
    loaders = [DataLoader(GeometricPairs(root, args.task), batch_size=1, shuffle=shuffle)
               for root, shuffle in ((args.train, True), (args.val, False))]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PairwiseHead(args.task, args.channels).to(device)
    initial_parameter = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    destination = Path(args.out)
    destination.mkdir(parents=True, exist_ok=True)
    logger = TrainingLogger(destination, args.task, model, optimizer, vars(args), device=str(device),
                            train_samples=len(loaders[0].dataset), validation_samples=len(loaders[1].dataset),
                            loss_definition='supervised pairwise objective; RGB quality metrics do not apply')
    best = float('inf')
    for epoch in range(args.epochs):
        metrics = []
        for training, loader in zip((True, False), loaders):
            model.train(training)
            total = 0.
            with torch.set_grad_enabled(training):
                for batch in loader:
                    loss = task_loss(model, {k: v.to(device) for k, v in batch.items()})
                    if not torch.isfinite(loss):
                        raise ValueError('Non-finite pairwise training loss')
                    if training:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                    total += loss.item()
            metrics.append(total / len(loader))
        logger.epoch(epoch + 1, optimizer, {'loss': metrics[0]}, {'loss': metrics[1]})
        if metrics[1] < best:
            best = metrics[1]
            torch.save(dict(model=model.state_dict(), task=args.task, contract='geometric_pairs_v1',
                            channels=args.channels, max_displacement=8., work_width=1024,
                            epoch=epoch + 1, val_loss=best,
                            train_metrics={'loss': metrics[0]}, validation_metrics={'loss': metrics[1]},
                            learning_rates=logger.learning_rates(optimizer),
                            parameter_change_l1=float((next(model.parameters()).detach() - initial_parameter).abs().sum())),
                       destination / f'{args.task}_best.pt')


if __name__ == '__main__':
    main()
