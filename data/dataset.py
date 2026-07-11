import torch
from torch.utils.data import Dataset

class PanoramaDataset(Dataset):
    def __init__(self, root, has_gt=False):
        self.root = root
        self.has_gt = has_gt
        self.length = 100  # placeholder

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        images = torch.randn(6, 3, 224, 224)      # N views
        rotations = torch.randn(6, 3, 3)          # SO(3)

        gt_panorama = None
        if self.has_gt:
            gt_panorama = torch.randn(3, 256, 512)

        return {
            "images": images,
            "rotations": rotations,
            "gt_panorama": gt_panorama
        }