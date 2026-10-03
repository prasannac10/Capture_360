"""Repeat one real training scene and export timing, VRAM and profiler evidence."""
import argparse
import csv
import hashlib
import json
import platform
import statistics
import subprocess
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import yaml

from .. import train_tiled
from ..data.tile_dataset import VariableTilePanoramaDataset
from ..data.native_rgb import NativeRGBSource
from ..models.panorama_model import PanoramaModel
from ..models import panorama_model
from ..models.tile_spherical import panorama_contract
from ..train.panorama_loss import PanoramaLoss
from ..utils.ema import EMA


def instrument(model, optimizer, criterion):
    """Name profiler regions without changing the production training code."""
    stack = ExitStack()

    def wrap(function, name):
        def measured(*args, **kwargs):
            with torch.profiler.record_function(name):
                return function(*args, **kwargs)
        return measured

    for obj, method, label in (
        (model, 'forward_scene', 'bench.forward'),
        (model.encoder, 'forward', 'bench.encoder'),
        (model.decoder, 'forward', 'bench.decoder'),
        (optimizer, 'step', 'bench.optimizer'),
        (criterion, 'forward', 'bench.loss'),
        (panorama_model, 'project_tile_features', 'bench.feature_projection'),
        (NativeRGBSource, '__call__', 'bench.native_rgb_loading_projection'),
        (NativeRGBSource, '_image', 'bench.native_photo_loading'),
        (torch.Tensor, 'backward', 'bench.backward'),
    ):
        stack.enter_context(patch.object(obj, method, wrap(getattr(obj, method), label)))
    original = train_tiled.iter_tile_batches

    def batches(*args, **kwargs):
        iterator = iter(original(*args, **kwargs))
        while True:
            with torch.profiler.record_function('bench.source_tile_loading'):
                try:
                    chunk = next(iterator)
                except StopIteration:
                    return
            yield chunk
    stack.enter_context(patch.object(train_tiled, 'iter_tile_batches', batches))
    return stack


class GPUUtilization:
    """Best-effort one-second nvidia-smi sampling, retaining every GPU UUID."""
    def __init__(self):
        self.samples, self.error = [], None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=uuid,name,utilization.gpu,memory.used',
                     '--format=csv,noheader,nounits'], capture_output=True, text=True,
                    timeout=5, check=True,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                for row in csv.reader(result.stdout.splitlines()):
                    uuid, name, utilization, memory = [value.strip() for value in row]
                    self.samples.append(dict(uuid=uuid, name=name, utilization_pct=float(utilization),
                                             memory_used_mib=float(memory), timestamp=time.time()))
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                self.error = str(error)
                return
            self.stop.wait(1)

    def summary(self):
        groups = {}
        for sample in self.samples:
            groups.setdefault(sample['uuid'], []).append(sample)
        return {uuid: dict(name=rows[0]['name'], samples=len(rows),
                           mean_utilization_pct=statistics.mean(r['utilization_pct'] for r in rows),
                           peak_utilization_pct=max(r['utilization_pct'] for r in rows),
                           peak_memory_used_mib=max(r['memory_used_mib'] for r in rows))
                for uuid, rows in groups.items()}


def fingerprint(sample, scene):
    digest = hashlib.sha256()
    for name in ('image_size', 'camera_params', 'poses'):
        digest.update(sample[name].numpy().tobytes())
    for path in [*sample['frame_paths'], str(scene / 'panorama.png')]:
        with open(path, 'rb') as source:
            while chunk := source.read(4 * 1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--config', default='panorama/stitching/config.yaml')
    parser.add_argument('--out', required=True, help='New output directory; existing reports are preserved')
    parser.add_argument('--steps', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--checkpoint', help='Identical initialization checkpoint for both GPUs')
    parser.add_argument('--allow-cpu', action='store_true', help='Synthetic execution checks only')
    args = parser.parse_args(argv)
    if args.steps < 1 or args.warmup < 0:
        parser.error('steps must be positive and warmup nonnegative')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not args.allow_cpu:
        parser.error('CUDA GPU required; --allow-cpu is only for execution tests')
    output = Path(args.out).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Choose an empty/new benchmark output directory')
    output.mkdir(parents=True, exist_ok=True)
    scene = Path(args.scene).resolve()
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    m, tr = config['model'], config['training']
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
    started = time.perf_counter()
    dataset = VariableTilePanoramaDataset(scene.parent, True, m['tile_size'], m['tile_overlap'],
                                          input_config=config['input'])
    index = next(i for i, path in enumerate(dataset.scenes) if path.resolve() == scene)
    sample = dataset[index]
    sample.update(_tile_size=m['tile_size'], _overlap=m['tile_overlap'],
                  _native_crop_size=tr.get('native_crop_size', 1024),
                  _seam_crop_probability=tr.get('seam_crop_probability', .5))
    load_seconds = time.perf_counter() - started
    scene_hash = fingerprint(sample, scene)
    detail = m.get('detail', {})
    model = PanoramaModel(m['feature_dim'], (m['pano_feature_height'], m['pano_feature_width']),
                          (m['train_output_height'], m['train_output_width']), m['tile_size'],
                          m['encoder']['backbone'], m['encoder']['pretrained'] and not args.checkpoint,
                          m['attention']['heads'], m['attention'].get('layers', 2),
                          tr.get('decoder_output_tile', 1024), detail.get('mode', 'features'),
                          detail.get('residual_scale', .1), tr.get('checkpoint_encoder', False)).to(device)
    checkpoint_hash = None
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get('contract') != panorama_contract(detail.get('mode', 'features')):
            raise ValueError('Benchmark checkpoint must match the model contract')
        if state.get('config', {}).get('model', {}).get('detail', {}) != detail:
            raise ValueError('Benchmark checkpoint detail settings must match the config')
        model.load_state_dict(state['model'], strict=True)
    model.train()
    code_hash = hashlib.sha256()
    code_root = Path(__file__).resolve().parents[2]
    for source in sorted(code_root.rglob('*.py')):
        code_hash.update(source.relative_to(code_root).as_posix().encode())
        code_hash.update(source.read_bytes())
    initial_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        initial_hash.update(name.encode())
        initial_hash.update(value.detach().cpu().numpy().tobytes())
    criterion = PanoramaLoss(config.get('loss', {})).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tr['lr'], weight_decay=tr['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and tr.get('mixed_precision', True))
    ema = EMA(model, tr['ema_decay']) if tr.get('use_ema', False) else None
    tile_batch = int(tr.get('tile_batch_size', 4))

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()

    def step(index):
        torch.manual_seed(args.seed + index)
        return train_tiled.run_scene(model, sample, device, tile_batch, True, optimizer,
                                    scaler, criterion, ema, return_metrics=True)

    for index in range(args.warmup):
        step(index)
    sync()
    telemetry = GPUUtilization()
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        telemetry.thread.start()
    timings, losses = [], []
    try:
        # Throughput is measured without the profiler or per-stage barriers.
        for index in range(args.steps):
            sync()
            started = time.perf_counter()
            result = step(args.warmup + index)
            sync()
            timings.append(time.perf_counter() - started)
            losses.append(result)
        peak_allocated = torch.cuda.max_memory_allocated() if device.type == 'cuda' else None
        peak_reserved = torch.cuda.max_memory_reserved() if device.type == 'cuda' else None
    finally:
        telemetry.stop.set()
        if device.type == 'cuda':
            telemetry.thread.join(timeout=6)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with instrument(model, optimizer, criterion), torch.profiler.profile(activities=activities) as profile:
        step(args.warmup + args.steps)
        sync()
    profile.export_chrome_trace(str(output / 'trace.json'))
    stages = {event.key: dict(calls=event.count, cpu_total_ms=event.cpu_time_total / 1000,
                              cpu_self_ms=event.self_cpu_time_total / 1000,
                              device_total_ms=getattr(event, 'device_time_total', 0) / 1000)
              for event in profile.key_averages() if event.key.startswith('bench.')}
    report = dict(status='complete', device=str(device), gpu_name=torch.cuda.get_device_name() if device.type == 'cuda' else None,
                  scene=str(scene), scene_sha256=scene_hash, checkpoint_sha256=checkpoint_hash,
                  code_sha256=code_hash.hexdigest(), initial_weights_sha256=initial_hash.hexdigest(),
                  torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                  platform=platform.platform(), config=config, seed=args.seed, warmup=args.warmup,
                  measured_steps=args.steps, setup_scene_loading_seconds=load_seconds,
                  step_seconds=timings, mean_step_seconds=statistics.mean(timings),
                  median_step_seconds=statistics.median(timings), step_metrics=losses,
                  peak_allocated_bytes=peak_allocated, peak_reserved_bytes=peak_reserved,
                  gpu_utilization=telemetry.summary(), telemetry_error=telemetry.error,
                  profiler_stages=stages,
                  notes=['One optimizer step is one crop, not one scene or epoch.',
                         'Profiler captures a separate step; inclusive stage timings overlap and must not be added.',
                         'Encoder timings include activation-checkpoint recomputation inside backward.',
                         'Scene reference/calibration/exposure loading is setup; repeated photo loading remains in each step.',
                         'GPU utilization is sampled by nvidia-smi across all visible physical GPUs; match GPU UUIDs.',
                         'Benchmark updates an in-memory model only; no training checkpoints are overwritten.'])
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(report=str(output / 'report.json'), mean_step_seconds=report['mean_step_seconds'],
                          peak_allocated_bytes=peak_allocated, gpu_name=report['gpu_name']), indent=2))
    return report


if __name__ == '__main__':
    main()
