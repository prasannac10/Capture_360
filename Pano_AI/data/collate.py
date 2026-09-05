import torch
from models.pose import normalize_pose


def panorama_collate_fn(batch):
    max_n = max(item["images"].shape[0] for item in batch)
    bsz = len(batch)
    _, channels, height, width = batch[0]["images"].shape
    images = torch.zeros(bsz, max_n, channels, height, width, dtype=batch[0]["images"].dtype)
    poses = torch.zeros(bsz, max_n, 3, dtype=torch.float32)
    rotations = torch.eye(3, dtype=torch.float32).view(1, 1, 3, 3).repeat(bsz, max_n, 1, 1)
    mask = torch.zeros(bsz, max_n, dtype=torch.bool)
    targets = []
    for i, item in enumerate(batch):
        n = item["images"].shape[0]
        images[i, :n] = item["images"]
        poses[i, :n] = item["poses"]
        rotations[i, :n] = normalize_pose(item["poses"])
        mask[i, :n] = True
        if item.get("gt_panorama") is not None:
            targets.append(item["gt_panorama"])
    result = {"images": images, "poses": poses, "rotations": rotations, "mask": mask, "scenes": [item["scene"] for item in batch]}
    result["gt_panorama"] = torch.stack(targets) if len(targets) == bsz else None
    return result
