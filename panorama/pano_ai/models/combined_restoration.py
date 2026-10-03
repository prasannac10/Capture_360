"""Spatial restoration from reviewed stitched/final panorama pairs."""
from torch import nn
from torch.nn import functional as F
from .restoration_backbone import RestorationUNet

COMBINED_CONTRACT = 'paired_panorama_restoration_v2'
LEGACY_COMBINED_CONTRACT = 'paired_panorama_restoration_v1'


class LegacyCombinedRestorationUNet(RestorationUNet):
    """Load existing v1 checkpoints without changing their architecture."""
    def __init__(self, channels=32):
        super().__init__(in_channels=3, out_channels=3, base_channels=channels)


class ContextResidual(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)

    def forward(self, x):
        return x + .1 * F.silu(self.conv(x))


class CombinedRestorationUNet(RestorationUNet):
    """Identity-initialized residual U-Net with wider, tile-local context.

    RGB residuals can learn deghosting, local seam cleanup, denoising and detail
    restoration as well as color, when those changes exist in aligned targets.
    No spatial normalization: tile statistics must not change output colors.
    """
    inference_halo = 160
    lattice_multiple = 8

    def __init__(self, channels=32):
        super().__init__(in_channels=3, out_channels=3, base_channels=channels)
        for block in (self.enc1, self.enc2, self.enc3, self.bottleneck, self.dec3, self.dec2, self.dec1):
            for index, layer in enumerate(block.block):
                if isinstance(layer, nn.GroupNorm):
                    block.block[index] = nn.Identity()
        self.bottleneck = nn.Sequential(self.bottleneck, *[
            ContextResidual(channels * 4, dilation) for dilation in (1, 2, 4)])
        # Begin as an exact identity, so an untrained residual cannot damage detail.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        height, width = x.shape[-2:]
        padded = F.pad(x, (0, (-width) % 8, 0, (-height) % 8), mode='replicate')
        return super().forward(padded)[..., :height, :width]
