"""Execution check for the simpler before/after correction route, not image quality."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
from .run_training import run
from ..highres_correction import HighResolutionCorrectionPipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(args.out).resolve()
    if root.exists() and any(root.iterdir()):
        raise ValueError('Use a new output folder to preserve previous results')
    root.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        split = 'train' if i < 2 else 'val'
        folder = root / 'dataset' / 'restoration' / split / f'synthetic_{i}'
        folder.mkdir(parents=True)
        target = np.random.default_rng(i).integers(30, 170, (64, 128, 3), dtype=np.uint8)
        # Exercise spatial repairs as well as photometry: an exposure step and
        # a locally doubled edge/texture region, with a known clean target.
        image = target.copy()
        image[:, 64:] = np.rint(image[:, 64:].astype(np.float32) * .75).astype(np.uint8)
        ghost = np.rint(.5 * target.astype(np.float32) + .5 * np.roll(target, 3, axis=1)).astype(np.uint8)
        image[20:44, 40:88] = ghost[20:44, 40:88]
        Image.fromarray(image).save(folder / 'before.png')
        Image.fromarray(target).save(folder / 'after.png')
        (folder / 'pair.json').write_text(json.dumps(dict(scene_id=f'synthetic_{i}', before='before.png', after='after.png',
            projection='equirectangular', alignment_verified=True)), encoding='utf-8')
    job = dict(dataset_root='dataset', output_dir='training',
               pipeline_config=str(Path(__file__).resolve().parents[2] / 'stitching' / 'config.yaml'),
               seed=42, cpu_threads=2, epochs=2, tasks=['combined'],
               combined=dict(input_size=[32, 32], crops_per_scene=4, base_channels=8, batch_size=1))
    job_path = root / 'job.yaml'
    job_path.write_text(yaml.safe_dump(job), encoding='utf-8')
    report = run(job_path)
    checkpoint = report['tasks']['combined']['checkpoint']
    if report['tasks']['combined']['parameter_change_l1'] <= 0:
        raise RuntimeError('No weight update observed')
    config = {'correction': {'toggles': {'combined': True}, 'checkpoints': {'combined': checkpoint}}}
    pipeline = HighResolutionCorrectionPipeline(config, job_path, 'cpu')
    output, metadata = pipeline.run(image, None, 32, 8, output_dir=root / 'inference')
    if output.shape != image.shape or metadata['applied_stages'] != ['combined']:
        raise RuntimeError('Combined checkpoint inference did not complete')
    Image.fromarray(image).save(root / 'inference' / 'before.png')
    Image.fromarray(target).save(root / 'inference' / 'target.png')
    report.update(status='passed', scenes=3, train_scenes=2, validation_scenes=1,
                   checkpoint_reload=True, mask_required=False, quality_claim='None: synthetic execution test only')
    (root / 'smoke_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(root / 'smoke_report.json')


if __name__ == '__main__':
    main()
