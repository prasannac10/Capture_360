"""Multi-scale panorama decoder producing the canonical 12K x 6K output."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.GELU(),nn.Conv2d(channels,channels,3,padding=1))
    def forward(self,x): return F.gelu(x+self.net(x))


class MultiScalePanoramaDecoder(nn.Module):
    def __init__(self, dim=64, output_channels=3, output_size=(6000,12000), refinement_blocks=3):
        super().__init__(); self.output_size=output_size
        hidden=max(32,dim)
        self.in_proj=nn.Sequential(nn.Conv2d(dim,hidden,3,padding=1),nn.GELU())
        self.refine=nn.Sequential(*[ResidualBlock(hidden) for _ in range(refinement_blocks)])
        self.detail=nn.Sequential(nn.Conv2d(hidden,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,output_channels,3,padding=1))

    def forward(self,x):
        x=self.refine(self.in_proj(x))
        x=F.interpolate(x,size=self.output_size,mode='bilinear',align_corners=False)
        return torch.sigmoid(self.detail(x))
