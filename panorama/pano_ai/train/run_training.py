"""Run the separately supervised stages from one local or S3 dataset root."""
import argparse
import copy
import json
from pathlib import Path
import random
import numpy as np
import torch
import yaml
from ..data.storage import materialize_dataset

TASKS = ('panorama', 'alignment', 'blending', 'glare', 'nadir_zenith', 'ghost_removal', 'color', 'combined')


def run(config_path):
    config_path = Path(config_path).resolve()
    job = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    base = config_path.parent
    def local(value):
        return (base / value).resolve()
    seed = int(job.get('seed', 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(job.get('cpu_threads', 4)))
    data = materialize_dataset(job['dataset_root'], local(job.get('cache_dir', 'dataset_cache')), base)
    output = local(job['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    tasks = job.get('tasks', ['combined'])
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(TASKS):
        raise ValueError('Specify distinct supported training tasks')
    shared = yaml.safe_load(local(job['pipeline_config']).read_text(encoding='utf-8'))
    report = dict(dataset_root=job['dataset_root'], local_dataset=str(data), seed=seed,
                  torch_version=str(torch.__version__), device='cuda' if torch.cuda.is_available() else 'cpu', tasks={})
    for task in tasks:
        print(f'\nTraining {task}', flush=True)
        options = job.get(task, {})
        if (options.get('resume') or options.get('finetune')) and task in ('panorama', 'alignment', 'blending'):
            raise ValueError('Resume/fine-tuning currently supports restoration tasks only')
        epochs = int(options.get('epochs', job.get('epochs', 1)))
        if epochs < 1:
            raise ValueError('Training epochs must be positive')
        task_out = output / task
        task_out.mkdir(parents=True, exist_ok=True)
        if task == 'panorama':
            from ..train_tiled import main
            config = copy.deepcopy(shared)
            config['model'].update(options.get('model', {}))
            config['training'].update(options.get('training', {}))
            config['training'].update(training_data=str(data / 'panorama' / 'train'),
                                      val_data=str(data / 'panorama' / 'val'), model_path=str(task_out), epochs=epochs)
            if not (data / 'panorama' / 'val').is_dir():
                raise ValueError('Missing panorama validation split')
            if 'loss' in options:
                config['loss'] = options['loss']
            if 'input' in options:
                config['input'] = options['input']
            written = task_out / 'training.yaml'
            written.write_text(yaml.safe_dump(config), encoding='utf-8')
            main(['--config', str(written)])
            checkpoint = task_out / config['training']['best_model_name']
        elif task in ('alignment', 'blending'):
            from .train_pairwise import main
            main(['--task', task, '--train', str(data / 'pairs' / task / 'train'),
                  '--val', str(data / 'pairs' / task / 'val'), '--out', str(task_out),
                  '--epochs', str(epochs), '--channels', str(options.get('channels', 32)),
                  '--lr', str(options.get('lr', 1e-4))])
            checkpoint = task_out / f'{task}_best.pt'
        else:
            from .train_restoration import main
            settings = dict(input_size=[shared['model']['tile_size']] * 2,
                            base_channels=32, batch_size=1, lr=1e-4)
            settings.update(options)
            settings['epochs'] = epochs
            for initialization in ('resume', 'finetune'):
                if settings.get(initialization):
                    settings[initialization] = str(local(settings[initialization]))
            written = task_out / 'training.yaml'
            written.write_text(yaml.safe_dump(settings), encoding='utf-8')
            arguments = ['--stage', task, '--data', str(data / 'restoration' / 'train'),
                         '--val-data', str(data / 'restoration' / 'val'), '--config', str(written), '--out', str(task_out)]
            mobile = job.get('mobile_validation', {})
            if task == 'combined' and mobile.get('enabled', False):
                arguments += ['--mobile-val-data', str(data / mobile.get('path', 'restoration/mobile_val'))]
            main(arguments)
            checkpoint = task_out / f'{task}_best.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        val_loss = float(state.get('val_loss', state.get('metrics', {}).get('loss', float('nan'))))
        if not np.isfinite(val_loss) or not all(torch.isfinite(t).all() for t in state['model'].values()):
            raise RuntimeError(f'{task}: non-finite checkpoint or validation loss')
        report['tasks'][task] = dict(checkpoint=str(checkpoint), val_loss=val_loss,
                                    training_log=str(task_out / 'training_log.jsonl'),
                                    train_metrics=state.get('train_metrics'),
                                    validation_metrics=state.get('validation_metrics', state.get('metrics')),
                                    learning_rates=state.get('learning_rates'),
                                    parameter_change_l1=state.get('parameter_change_l1'), epoch=state['epoch'],
                                    checkpoint_selection=state.get('checkpoint_selection'),
                                    finetuned_from=state.get('finetuned_from'), resumed_from=state.get('resumed_from'),
                                    selection_loss=state.get('selection_loss'), mobile_metrics=state.get('mobile_metrics'),
                                    paired_baseline=state.get('paired_baseline'),
                                    paired_metrics=state.get('metrics'),
                                    paired_beats_baseline_l1=state.get('paired_beats_baseline_l1'),
                                    mobile_baseline=state.get('mobile_baseline'),
                                    mobile_beats_baseline_l1=state.get('mobile_beats_baseline_l1'))
        (output / 'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.config), indent=2))


if __name__ == '__main__':
    main()
