"""Native-resolution tile encoder. Output stride is explicitly 8."""

import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, dim=64, backbone="resnet18", pretrained=True):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"Unsupported encoder backbone: {backbone}")
        from torchvision.models import ResNet18_Weights, resnet18

        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(
            base.conv1, base.bn1, base.relu, base.maxpool, base.layer1, base.layer2
        )
        self.projection = nn.Sequential(
            nn.Conv2d(128, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.GELU()
        )
        self.feature_stride = 8

    def forward(self, x):
        return self.projection(self.backbone(x))
