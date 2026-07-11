import torch
import torch.nn as nn
from spherical import spherical_project

class PanoramaModel(nn.Module):
    def __init__(self, encoder, aggregator, decoder, pano_h, pano_w):
        super().__init__()
        self.encoder = encoder
        self.aggregator = aggregator
        self.decoder = decoder
        self.pano_h = pano_h
        self.pano_w = pano_w

        self.last_spherical = None

    def forward(self, images, rotations):
        feats = self.encoder(images)              # [N, C]
        ctx = self.aggregator(feats)               # [C]

        pano_feat = spherical_project(
            feats, rotations, self.pano_h, self.pano_w
        )

        self.last_spherical = pano_feat

        pano_feat = pano_feat * torch.sigmoid(ctx[:, None, None])
        return self.decoder(pano_feat)