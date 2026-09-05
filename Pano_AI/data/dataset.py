from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

INPUT_SIZE = (224, 224)
PANO_SIZE = (256, 512)


def resize_letterbox(image: Image.Image, size=INPUT_SIZE):
    image = image.convert("RGB")
    target_h, target_w = size
    src_w, src_h = image.size
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = max(1, round(src_w * scale)), max(1, round(src_h * scale))
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
    canvas.paste(image, ((target_w - new_w) // 2, (target_h - new_h) // 2))
    return torch.from_numpy(np.asarray(canvas)).permute(2, 0, 1).float() / 255.0


def load_panorama(path: Path):
    image = Image.open(path).convert("RGB").resize((PANO_SIZE[1], PANO_SIZE[0]), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float() / 255.0


class PanoramaDataset(Dataset):
    """Real scene loader: images/*.png + poses.pt + optional panorama.png."""
    def __init__(self, root, has_gt=True):
        self.root = Path(root)
        self.has_gt = has_gt
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        self.scenes = sorted(p for p in self.root.iterdir() if p.is_dir() and (p / "images").is_dir())
        if not self.scenes:
            raise RuntimeError(f"No scene directories found under {self.root}")

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        scene = self.scenes[idx]
        image_files = sorted((scene / "images").glob("*.png"))
        if not image_files:
            raise RuntimeError(f"No PNG frames found in {scene / 'images'}")
        pose_path = scene / "poses.pt"
        if not pose_path.exists():
            raise FileNotFoundError(f"Missing poses.pt in {scene}")
        poses = torch.load(pose_path, map_location="cpu", weights_only=False)
        if isinstance(poses, dict):
            for key in ("poses", "ypr", "rotations"):
                if key in poses:
                    poses = poses[key]
                    break
        poses = torch.as_tensor(poses, dtype=torch.float32)
        if poses.ndim != 2 or poses.shape[1] != 3:
            raise ValueError(f"{pose_path} must contain [N,3] yaw/pitch/roll degrees; got {poses.shape}")
        if len(image_files) != poses.shape[0]:
            raise ValueError(f"{scene}: {len(image_files)} images but {poses.shape[0]} poses")
        images = torch.stack([resize_letterbox(Image.open(path)) for path in image_files])
        sample = {"images": images, "poses": poses, "scene": scene.name}
        if self.has_gt:
            gt_path = scene / "panorama.png"
            if not gt_path.exists():
                raise FileNotFoundError(f"Supervised scene is missing panorama.png: {scene}")
            sample["gt_panorama"] = load_panorama(gt_path)
        else:
            sample["gt_panorama"] = None
        return sample
