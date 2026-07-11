import torch

def panorama_collate_fn(batch):
    images_list, poses_list, targets = zip(*batch)

    B = len(images_list)
    max_n = max(x.shape[0] for x in images_list)

    _, C, H, W = images_list[0].shape

    images_pad = torch.zeros(B, max_n, C, H, W)
    poses_pad = torch.zeros(B, max_n, 2)
    mask = torch.zeros(B, max_n, dtype=torch.bool)

    for i in range(B):
        n = images_list[i].shape[0]
        images_pad[i, :n] = images_list[i]
        poses_pad[i, :n] = poses_list[i]
        mask[i, :n] = True

    targets = torch.stack(targets)
    return images_pad, poses_pad, mask, targets
