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
        self.checkpoint_gradients = False
        self.register_buffer('input_mean', torch.tensor([.485, .456, .406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('input_std', torch.tensor([.229, .224, .225]).view(1, 3, 1, 1), persistent=False)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        # Small tile batches vary greatly in sky/ground content. Keep pretrained
        # running statistics fixed across tiles and across the two encoder passes.
        # Affine BN parameters still receive gradients.
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        return self

    def forward(self, x):
        normalized = (x - self.input_mean) / self.input_std
        if self.checkpoint_gradients and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(lambda value: self.projection(self.backbone(value)),
                              normalized, use_reentrant=False)
        return self.projection(self.backbone(normalized))
