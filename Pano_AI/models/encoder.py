"""Spatial image encoder for the multi-view panorama model."""

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    """Encode each input view while preserving spatial correspondence.

    A torchvision ResNet18 can be used as the feature extractor. The classifier
    and global pooling are removed, so the output remains a spatial feature map.
    ``pretrained=False`` keeps tests/offline development deterministic.
    """

    def __init__(self, dim: int, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"Unsupported encoder backbone: {backbone}")

        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("torchvision is required for ImageEncoder") from exc

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base = resnet18(weights=weights)
        self.backbone = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool,
            base.layer1,
            base.layer2,
            base.layer3,
            base.layer4,
        )
        self.projection = nn.Sequential(
            nn.Conv2d(512, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(x))
