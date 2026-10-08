"""Flushed console summaries and append-only structured training history."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid


class TrainingLogger:
    def __init__(self, output_dir, task, model, optimizer, config, **details):
        self.path = Path(output_dir) / 'training_log.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.task = task
        self.run_id = uuid.uuid4().hex
        self.started = time.monotonic()
        self.write('start', optimizer=type(optimizer).__name__,
                   learning_rates=self.learning_rates(optimizer),
                   optimizer_groups=[{key: group[key] for key in ('lr', 'weight_decay', 'betas', 'eps')
                                      if key in group} for group in optimizer.param_groups],
                   parameters_total=sum(p.numel() for p in model.parameters()),
                   parameters_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   config=config, metric_definitions={
                       'pixel_accuracy_pct': 'Percent of pixels with all RGB channel errors <= 8/255; not classification accuracy',
                       'psnr': 'Mean per-image PSNR in dB, capped at 99 for exact matches',
                       'ssim': 'Mean per-image global SSIM proxy, not windowed SSIM',
                       'l1': 'Mean absolute RGB error in [0,1]',
                       'seam_l1': 'RGB reference error in observed ownership boundary bands',
                       'seam_gradient_l1': 'Reference gradient error within ownership boundary bands',
                       'seam_pixels': 'Evaluated seam pixels; zero means no seam observed'}, **details)

    @staticmethod
    def learning_rates(optimizer):
        return [float(group['lr']) for group in optimizer.param_groups]

    def write(self, event, **values):
        row = dict(event=event, task=self.task, run_id=self.run_id,
                   timestamp_utc=datetime.now(timezone.utc).isoformat(), **values)
        payload = json.dumps(row, allow_nan=False, default=str)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(payload + '\n')
        if event == 'start':
            print(f'[{self.task}] training_start {payload}', flush=True)
        return row

    def epoch(self, epoch, optimizer, train, validation=None, mobile=None, **details):
        lr = self.learning_rates(optimizer)
        row = self.write('epoch', epoch=epoch, learning_rates=lr, train=train,
                         validation=validation, mobile=mobile,
                         elapsed_seconds=time.monotonic()-self.started, **details)
        fields = [f'[{self.task}] epoch={epoch}', 'lr=' + ','.join(f'{value:.6g}' for value in lr)]
        for name, metrics in (('train', train), ('val', validation), ('mobile', mobile)):
            if metrics is None:
                continue
            for key in ('loss', 'l1', 'pixel_accuracy_pct', 'psnr', 'ssim', 'seam_l1', 'seam_gradient_l1', 'seam_pixels'):
                if key in metrics:
                    label = 'ssim_global' if key == 'ssim' else key
                    fields.append(f'{name}_{label}={metrics[key]:.6f}')
        print(' | '.join(fields), flush=True)
        return row
