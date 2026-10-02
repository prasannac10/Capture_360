"""One RGB residual model trained from reviewed stitched/final panorama pairs."""
from .restoration_backbone import RestorationUNet


class CombinedRestorationUNet(RestorationUNet):
    def __init__(self, channels=32):
        super().__init__(in_channels=3, out_channels=3, base_channels=channels)
