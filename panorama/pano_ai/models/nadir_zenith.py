"""Mask-aware learned nadir/zenith inpainting stage."""

from .restoration_backbone import RestorationUNet


class NadirZenithInpainter(RestorationUNet):
    """RGB + correction-mask U-Net; mask value 1 marks pixels to repair."""

    def __init__(self, channels: int = 32):
        super().__init__(in_channels=3, out_channels=3, base_channels=channels, mask_channels=1)

    def forward(self, image, mask):
        return super().forward(image, mask)
