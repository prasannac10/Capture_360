"""High-resolution shared tile encoder.

The encoder consumes original-resolution tiles rather than resizing a complete
camera frame to a tiny image. It emits a spatial feature map plus a compact
token for global attention.
"""
import torch
import torch.nn as nn


class HighResTileEncoder(nn.Module):
    def __init__(self, dim: int = 192, backbone: str = "resnet18", pretrained: bool = True):
        super().__init__()
        if backbone != "resnet18":
            raise ValueError(f"Unsupported tile encoder backbone: {backbone}")
        from torchvision.models import ResNet18_Weights, resnet18
        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool,
                                      base.layer1, base.layer2, base.layer3, base.layer4)
        self.proj = nn.Sequential(nn.Conv2d(512, dim, 1, bias=False), nn.BatchNorm2d(dim), nn.GELU())
        self.token_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        feat = self.proj(self.backbone(x))
        token = self.token_pool(feat).flatten(1)
        return feat, token
