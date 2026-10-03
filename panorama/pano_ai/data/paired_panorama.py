"""Reviewed panorama pairs, sampled at native scale without defect labels."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


class PairedPanoramaDataset(Dataset):
    def __init__(self, root, input_size=(512, 512), crops_per_scene=12, random_crops=False, required_domain=None,
                 changed_crop_probability=.5):
        self.size = tuple(input_size)
        self.crops_per_scene = int(crops_per_scene)
        self.random_crops = random_crops
        self.changed_crop_probability = float(changed_crop_probability)
        if not 0 <= self.changed_crop_probability <= 1:
            raise ValueError('changed_crop_probability must be in [0,1]')
        self._change_maps = {}
        if len(self.size) != 2 or min(self.size) < 8 or self.crops_per_scene < 1:
            raise ValueError('Require positive crops_per_scene and crop dimensions >=8')
        self.pairs, self.scene_ids = [], set()
        self.source_scene_ids, self.records = set(), []
        for manifest in sorted(Path(root).glob('*/pair.json')):
            record = json.loads(manifest.read_text(encoding='utf-8'))
            if required_domain is not None and record.get('domain') != required_domain:
                raise ValueError(f'{manifest}: validation requires domain={required_domain}; synthetic data is not real mobile validation')
            if record.get('alignment_verified') is not True or record.get('projection') != 'equirectangular':
                raise ValueError(f'{manifest}: review alignment and declare equirectangular projection first')
            scene_id = record['scene_id']
            if not isinstance(scene_id, str) or not scene_id or scene_id in self.scene_ids:
                raise ValueError('Each pair needs a distinct nonempty scene_id')
            paths = [(manifest.parent / record[key].replace('\\', '/')).resolve() for key in ('before', 'after')]
            dimensions = []
            for path in paths:
                with Image.open(path) as im:
                    dimensions.append(im.size)
                    if im.getexif().get(274, 1) != 1:
                        raise ValueError(f'{path}: normalize panorama orientation before pairing')
            w, h = dimensions[0]
            if dimensions[0] != dimensions[1] or w != 2 * h:
                raise ValueError('Reviewed pairs must have identical full-sphere 2:1 dimensions')
            if h < self.size[0] or w < self.size[1]:
                raise ValueError('Crop exceeds native panorama dimensions')
            self.pairs.append((paths, (w, h), scene_id))
            self.scene_ids.add(scene_id)
            source_id = record.get('source_scene_id', scene_id)
            if not isinstance(source_id, str) or not source_id:
                raise ValueError('source_scene_id must identify the physical scene')
            self.source_scene_ids.add(source_id)
            self.records.append(record)
        if not self.pairs:
            raise ValueError(f'No reviewed panorama pairs found under {root}; expected <scene>/pair.json')

    def __len__(self):
        return len(self.pairs) * self.crops_per_scene

    def _change_map(self, paths, scene_id):
        if scene_id not in self._change_maps:
            previews = []
            for path in paths:
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    image.thumbnail((256, 128), Image.Resampling.BILINEAR)
                    previews.append(np.asarray(image, dtype=np.float32) / 255.)
            difference = previews[1] - previews[0]
            # Remove a global color offset so it does not dominate local defect sampling.
            difference -= np.median(difference, axis=(0, 1), keepdims=True)
            importance = np.maximum(np.abs(difference).mean(-1) - .02, 0)
            self._change_maps[scene_id] = torch.from_numpy(importance.copy())
        return self._change_maps[scene_id]

    def __getitem__(self, index):
        paths, (w, h), scene_id = self.pairs[index // self.crops_per_scene]
        crop_index = index % self.crops_per_scene
        ch, cw = self.size
        columns = min(4, self.crops_per_scene)
        rows = (self.crops_per_scene + columns - 1) // columns
        y = round((h - ch) * (crop_index // columns) / max(1, rows - 1))
        # First column straddles the longitude join; no resize or inferred masks.
        x = (round(w * (crop_index % columns) / columns) - cw // 2) % w
        if self.random_crops:
            y = int(torch.randint(h - ch + 1, (1,)))
            x = int(torch.randint(w, (1,)))
            if float(torch.rand(())) < self.changed_crop_probability:
                importance = self._change_map(paths, scene_id)
                if importance.sum() > 0:
                    selected = int(torch.multinomial(importance.flatten(), 1))
                    row, column = divmod(selected, importance.shape[1])
                    y = max(0, min(h-ch, round((row+.5) * h / importance.shape[0] - ch/2)))
                    x = round((column+.5) * w / importance.shape[1] - cw/2) % w
        def read(path):
            with Image.open(path) as im:
                first_width = min(cw, w - x)
                patch = Image.new('RGB', (cw, ch))
                patch.paste(im.crop((x, y, x + first_width, y + ch)).convert('RGB'), (0, 0))
                if first_width < cw:
                    patch.paste(im.crop((0, y, cw - first_width, y + ch)).convert('RGB'), (first_width, 0))
                return torch.from_numpy(np.array(patch, copy=True)).permute(2, 0, 1).float() / 255
        return dict(input=read(paths[0]), target=read(paths[1]), scene_id=scene_id, stage='combined',
                    longitude_join_column=w-x if x+cw > w else -1)
