import torch.nn.functional as F

def supervised_loss(pred, gt, l1_weight=1.0):
    return l1_weight * F.l1_loss(pred, gt)