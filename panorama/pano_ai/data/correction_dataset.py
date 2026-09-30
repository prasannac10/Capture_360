from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

STAGES = ("glare", "dots", "nadir_zenith", "ghost_removal", "color")


def _tensor(path, size=None):
    im = Image.open(path).convert("RGB")
    if size:
        im = im.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.array(im, copy=True)).permute(2, 0, 1).float() / 255


class CorrectionPairDataset(Dataset):
    """Indexes paired before/after stage images when intermediate artifacts exist.
    Expected scene layout: stages/<stage>/before.png and after.png, or before_<stage>.png/after_<stage>.png.
    """

    def __init__(self, root, stage, input_size=(256, 512)):
        if stage not in STAGES:
            raise ValueError(stage)
        self.items = []
        root = Path(root)
        for scene in sorted(p for p in root.iterdir() if p.is_dir()):
            candidates = [
                (
                    scene / "stages" / stage / "before.png",
                    scene / "stages" / stage / "after.png",
                ),
                (scene / f"before_{stage}.png", scene / f"after_{stage}.png"),
            ]
            for before, after in candidates:
                if before.exists() and after.exists():
                    self.items.append((before, after))
                    break
        self.size = input_size
        self.stage = stage

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        a, b = self.items[i]
        before, after = _tensor(a), _tensor(b)
        if before.shape != after.shape:
            raise ValueError('Restoration pairs must share native coordinates')
        mask_path = a.with_name('mask.png') if a.name == 'before.png' else a.with_name(f'mask_{self.stage}.png')
        mask = None
        if self.stage != 'color':
            if not mask_path.exists():
                raise ValueError(f'Explicit restoration mask required: {mask_path}')
            with Image.open(mask_path) as image:
                mask = torch.from_numpy(np.array(image.convert('L'), copy=True)).float()[None] / 255
            if mask.shape[-2:] != before.shape[-2:]:
                raise ValueError('Restoration mask must use native image coordinates')
        h, w = before.shape[-2:]
        ch, cw = self.size or (h, w)
        if ch > h or cw > w:
            raise ValueError('Training crop exceeds native image size; do not rescale restoration texture')
        # Deterministic native crop; include the longitude join in alternate samples.
        y = (h - ch) // 2
        x = (w - cw) // 2 if i % 2 == 0 else w - cw // 2
        if mask is not None and mask.any():
            points = torch.nonzero(mask[0] > 0)
            py, px = points[len(points) // 2].tolist()
            y, x = min(max(py - ch // 2, 0), h - ch), (px - cw // 2) % w
        columns = torch.arange(x, x + cw) % w
        def crop(t):
            return t[:, y:y + ch, columns]
        result = {'input': crop(before), 'target': crop(after), 'stage': self.stage}
        if mask is not None:
            result['mask'] = crop(mask)
        return result
