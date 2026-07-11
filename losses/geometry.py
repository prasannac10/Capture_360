def geometry_loss(pano):
    return pano.abs().mean()