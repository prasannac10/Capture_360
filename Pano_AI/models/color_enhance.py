"""Learned color/illumination correction stage."""

from models.restoration_backbone import RestorationUNet


class ColorEnhancementUNet(RestorationUNet):
    def __init__(self, channels: int = 32):
        super().__init__(in_channels=3, out_channels=3, base_channels=channels)
