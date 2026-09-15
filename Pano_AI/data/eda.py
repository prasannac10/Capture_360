from pathlib import Path
from PIL import Image, ImageChops, ImageDraw
import json

def inspect_dataset(root):
    root = Path(root)
    scenes = sorted(p for p in root.iterdir() if p.is_dir() and (p / 'images').is_dir()) if root.exists() else []
    rows = []
    for scene in scenes:
        raws = sorted((scene / 'images').glob('*'))
        rows.append({'name': scene.name, 'raw_images': len(raws), 'has_panorama': (scene/'panorama.png').exists(), 'has_poses': (scene/'poses.pt').exists(), 'pts_files': [p.name for p in scene.glob('*.pts')], 'intermediate_files': [str(p.relative_to(scene)) for p in scene.rglob('*') if p.is_file() and p.name not in {'panorama.png','poses.pt'} and p.parent.name != 'images' and p.suffix.lower() != '.pts']})
    return {'root': str(root), 'scene_count': len(scenes), 'scenes': rows}

def save_before_after(before_path, after_path, output_path, label='before / after'):
    before = Image.open(before_path).convert('RGB'); after = Image.open(after_path).convert('RGB')
    h = max(before.height, after.height)
    before = before.resize((round(before.width*h/before.height), h)); after = after.resize((round(after.width*h/after.height), h))
    canvas = Image.new('RGB', (before.width+after.width, h+40), 'white')
    canvas.paste(before, (0,40)); canvas.paste(after, (before.width,40)); ImageDraw.Draw(canvas).text((10,10), label, fill='black'); canvas.save(output_path)

def difference_image(before_path, after_path, output_path):
    before = Image.open(before_path).convert('RGB'); after = Image.open(after_path).convert('RGB').resize(before.size)
    ImageChops.difference(before, after).save(output_path)

if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('root'); p.add_argument('--out', default='dataset_inventory.json'); a=p.parse_args()
    Path(a.out).write_text(json.dumps(inspect_dataset(a.root), indent=2), encoding='utf-8'); print('Wrote', a.out)
