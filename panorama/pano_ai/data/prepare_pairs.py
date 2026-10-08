"""Package reviewed before/after panoramas into a portable training bundle."""
import argparse
import json
from pathlib import Path
import shutil
from PIL import Image


def prepare(manifest, output, crop_size=1024, require_ai_outputs=False):
    manifest, output = Path(manifest).resolve(), Path(output).resolve()
    data = json.loads(manifest.read_text(encoding='utf-8'))
    if data.get('version') != 1 or not data.get('pairs') or crop_size < 8:
        raise ValueError('Expected version=1, nonempty pairs list and crop_size >=8')
    groups, ids, prepared = {}, set(), []
    for row in data['pairs']:
        if require_ai_outputs:
            if row.get('before_source') != 'ai_stitcher' or not row.get('stitcher_checkpoint'):
                raise ValueError('AI correction pairs require before_source=ai_stitcher and stitcher_checkpoint provenance')
            if Path(row['before']).name.lower() in ('stitched.jpg', 'edited.jpg'):
                raise ValueError('Use actual initial AI outputs, not PTGui reference images')
        name = row['scene_id']
        if not isinstance(name, str) or not name or not all(c.isalnum() or c in '_-' for c in name) or name in ids:
            raise ValueError('scene_id must be a unique safe folder name')
        ids.add(name)
        group = row.get('site', row.get('source_scene_id', name))
        split = row['split']
        if not isinstance(group, str) or not group:
            raise ValueError('source_scene_id must identify the physical location')
        if split not in ('train', 'val', 'mobile_val'):
            raise ValueError('split must be train, val or mobile_val')
        if row.get('domain') not in ('dslr', 'drone', 'real_mobile', 'synthetic_mobile'):
            raise ValueError('Declare domain: dslr, drone, real_mobile or synthetic_mobile')
        if split == 'mobile_val' and row['domain'] != 'real_mobile':
            raise ValueError('mobile_val accepts only real_mobile pairs')
        if group in groups and groups[group] != split:
            raise ValueError('Physical scene must stay in a single split')
        groups[group] = split
        if row.get('alignment_verified') is not True or row.get('projection') != 'equirectangular':
            raise ValueError('Review pair alignment and declare equirectangular projection before packaging')
        paths, sizes = [], []
        for key in ('before', 'after'):
            path = (manifest.parent / row[key].replace('\\', '/')).resolve()
            with Image.open(path) as im:
                sizes.append(im.size)
                if im.getexif().get(274, 1) != 1:
                    raise ValueError('Normalize image orientation before packaging')
            paths.append(path)
        w, h = sizes[0]
        if sizes[0] != sizes[1] or w != 2*h or min(w, h) < crop_size:
            raise ValueError('Pairs must have matching full-sphere 2:1 dimensions, at least crop_size high')
        prepared.append((row, paths, group))
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output folder; existing datasets are preserved')
    output.mkdir(parents=True, exist_ok=True)
    report = dict(version=1, source_manifest=str(manifest), alignment_check='user-reviewed; dimensions/orientation checked automatically', counts={})
    for row, paths, group in prepared:
        folder = output / 'restoration' / row['split'] / row['scene_id']
        folder.mkdir(parents=True)
        record = {k: row[k] for k in ('scene_id', 'domain', 'projection', 'alignment_verified')}
        record['source_scene_id'] = group
        for key in ('before_source', 'stitcher_checkpoint', 'site'):
            if key in row:
                record[key] = row[key]
        for key, path in zip(('before', 'after'), paths):
            filename = key + path.suffix.lower()
            shutil.copy2(path, folder / filename)
            record[key] = filename
        (folder / 'pair.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
        report['counts'][row['split']] = report['counts'].get(row['split'], 0) + 1
    (output / 'preparation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--crop-size', type=int, default=1024)
    parser.add_argument('--require-ai-outputs', action='store_true', help='Require AI stitcher provenance for correction training')
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.out, args.crop_size, args.require_ai_outputs), indent=2))


if __name__ == '__main__':
    main()
