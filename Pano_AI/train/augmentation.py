import torch

def equirectangular_augment(x, horizontal_flip=True):
    """Wrap-aware augmentation: horizontal roll and optional reflection."""
    if torch.rand(()) < .5:
        x=torch.roll(x, shifts=int(torch.randint(0,x.shape[-1],()).item()), dims=-1)
    if horizontal_flip and torch.rand(()) < .5:
        x=torch.flip(x,dims=(-1,))
    return x
