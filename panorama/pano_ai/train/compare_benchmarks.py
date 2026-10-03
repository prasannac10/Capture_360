"""Compare matching GPU benchmark reports, refusing incompatible workloads."""
import argparse
import json
from pathlib import Path


def compare(first, second):
    for key in ('scene_sha256', 'code_sha256', 'initial_weights_sha256', 'checkpoint_sha256',
                'seed', 'warmup', 'measured_steps'):
        if first[key] != second[key]:
            raise ValueError(f'Benchmarks differ in {key}; rerun with the same workload')
    for key in ('model', 'input', 'loss'):
        if first['config'].get(key) != second['config'].get(key):
            raise ValueError(f'Benchmarks differ in {key} configuration')
    irrelevant = {'training_data', 'val_data', 'model_path'}
    training = lambda report: {k: v for k, v in report['config']['training'].items() if k not in irrelevant}
    if training(first) != training(second):
        raise ValueError('Benchmarks differ in training configuration')
    return dict(first_gpu=first['gpu_name'], second_gpu=second['gpu_name'],
                first_mean_step_seconds=first['mean_step_seconds'],
                second_mean_step_seconds=second['mean_step_seconds'],
                measured_step_speedup=first['mean_step_seconds'] / second['mean_step_seconds'],
                first_peak_allocated_bytes=first['peak_allocated_bytes'],
                second_peak_allocated_bytes=second['peak_allocated_bytes'],
                first_gpu_utilization=first['gpu_utilization'],
                second_gpu_utilization=second['gpu_utilization'],
                first_profiler_stages=first['profiler_stages'],
                second_profiler_stages=second['profiler_stages'],
                software_matches=(first['torch_version'], first['cuda_version']) ==
                                 (second['torch_version'], second['cuda_version']),
                note='Measured optimizer-step speedup excludes setup, validation and checkpoint saving; it is not an epoch/runtime guarantee.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first', required=True)
    parser.add_argument('--second', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = compare(json.loads(Path(args.first).read_text()), json.loads(Path(args.second).read_text()))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
