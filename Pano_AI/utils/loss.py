import torch.nn.functional as F

def panorama_loss(pred, target):
    return F.l1_loss(pred, target)
