"""Generate reviewable correction-training inputs from Files/year/location/scene."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import yaml


def scan(root, out, domain=None):
    root, out = Path(root).resolve(), Path(out).resolve()
    if not root.is_dir():
        raise ValueError(f'Input directory does not exist: {root}')
    if domain not in (None, 'dslr', 'drone'):
        raise ValueError('Domain must be dslr or drone')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output folder; existing files are preserved')
    scenes, incomplete = [], []
    # Only scene folders with an images directory belong to this input layout.
    for images in sorted(root.rglob('images')):
        if not images.is_dir():
            continue
        scene = images.parent
        files = {p.name.lower(): p for p in scene.iterdir() if p.is_file()}
        missing = [name for name in ('stitched.jpg', 'edited.jpg') if name not in files]
        if missing:
            incomplete.append(dict(scene=str(scene), missing=missing))
            continue
        scenes.append((scene, files))
    if incomplete:
        raise ValueError('Incomplete scenes: ' + json.dumps(incomplete))
    if not scenes:
        raise ValueError('No scenes with images/, Stitched.jpg and Edited.jpg found')
    # Use the physical location name across years so repeat captures stay together.
    locations = sorted({scene.parent.name.casefold() for scene, _ in scenes})
    val_count = max(1, len(locations) // 5) if len(locations) > 1 else 0
    val_locations = set(locations[-val_count:]) if val_count else set()
    pairs = []
    for scene, files in scenes:
        label = re.sub(r'[^a-zA-Z0-9_-]', '_', '_'.join(scene.parts[-3:]))
        digest = hashlib.sha256(scene.as_posix().encode()).hexdigest()[:12]
        location = scene.parent.name.casefold()
        pairs.append(dict(scene_id=f'{label}_{digest}', source_scene_id=location,
                          split='val' if location in val_locations else 'train',
                          domain=domain, before=files['stitched.jpg'].as_posix(),
                          after=files['edited.jpg'].as_posix(),
                          projection='equirectangular', alignment_verified=False))
    template = Path(__file__).resolve().parents[1] / 'train/configs/training_job.yaml'
    job = yaml.safe_load(template.read_text(encoding='utf-8'))
    job.update(dataset_root='./training_bundle', output_dir='./training_run',
               cache_dir='./dataset_cache', tasks=['combined'],
               pipeline_config=(template.parent / job['pipeline_config']).resolve().as_posix())
    job['mobile_validation'] = dict(enabled=False, path='restoration/mobile_val')
    # A generated job always starts a new run.
    job['combined'].pop('resume', None)
    job['combined'].pop('finetune', None)
    report = dict(input_root=root.as_posix(), scene_count=len(pairs),
                  location_count=len(locations), validation_locations=sorted(val_locations),
                  needs_independent_validation=not bool(val_locations),
                  review_required=['Set domain per scene to dslr or drone',
                                   'Verify full-sphere 2:1 images and matching alignment',
                                   'Set alignment_verified=true only after review',
                                   'Review location identities and train/val split'])
    out.mkdir(parents=True, exist_ok=True)
    (out / 'pairs.json').write_text(json.dumps(dict(version=1, pairs=pairs), indent=2), encoding='utf-8')
    (out / 'training_job.yaml').write_text(yaml.safe_dump(job, sort_keys=False), encoding='utf-8')
    (out / 'scan_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Files, Files/year or Files/year/location')
    parser.add_argument('--out', default='outputs/sample_training_setup')
    parser.add_argument('--domain', choices=('dslr', 'drone'), help='Only if every scene has this source')
    args = parser.parse_args()
    print(json.dumps(scan(args.input, args.out, args.domain), indent=2))


if __name__ == '__main__':
    main()
