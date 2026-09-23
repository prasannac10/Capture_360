"""Variable-resolution panorama model with re-iterable tile streaming."""

import torch
import torch.nn as nn
from .encoder import ImageEncoder
from .tile_metadata import TileMetadataEncoder
from .tile_aggregator import TileAttentionAggregator, SceneToken
from .tile_spherical import project_tile_features, ypr_to_rot
from .decoder import MultiScalePanoramaDecoder


class PanoramaModel(nn.Module):
    def __init__(
        self,
        feature_dim=64,
        pano_feature_size=(750, 1500),
        output_size=(6000, 12000),
        tile_size=1024,
        backbone="resnet18",
        pretrained=True,
        attention_heads=8,
        attention_layers=2,
        output_tile=1024,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.pano_feature_size = pano_feature_size
        self.encoder = ImageEncoder(feature_dim, backbone, pretrained)
        self.metadata = TileMetadataEncoder(feature_dim)
        self.aggregator = TileAttentionAggregator(
            feature_dim, attention_heads, attention_layers
        )
        self.scene_token = SceneToken(feature_dim, attention_heads)
        self.condition = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.Sigmoid()
        )
        self.decoder = MultiScalePanoramaDecoder(
            feature_dim, 3, output_size, output_tile=output_tile
        )

    def _meta(self, xy, wh, size, cam, rot):
        s = size.view(1, 1, 2).clamp_min(1)
        xy = xy.view(1, -1, 2) / s
        wh = wh.view(1, -1, 2) / s
        image = torch.ones_like(xy)
        c = cam.view(1, 1, 6).expand(1, xy.shape[1], 6).clone()
        c[..., :2] /= s
        c[..., 2:4] /= s
        c[..., 5] /= 180.0
        r = rot.view(1, 1, 3, 3).expand(1, xy.shape[1], 3, 3)
        return self.metadata(xy, wh, image, c, r)

    def forward_scene(
        self, batch_factory, image_size, camera_params, poses, tile_batch_size=4
    ):
        """batch_factory() must yield (frame_index, tiles[K,3,H,W], xy[K,2], wh[K,2]). It is called twice."""
        dev = next(self.parameters()).device
        tokens = []
        count = 0
        for fi, tiles, xy, wh in batch_factory():
            for st in range(0, tiles.shape[0], tile_batch_size):
                x = tiles[st : st + tile_batch_size].to(dev)
                rot = ypr_to_rot(poses[fi : fi + 1].to(dev))
                feat = self.encoder(x)
                tokens.append(
                    feat.mean((-1, -2))
                    + self._meta(
                        xy[st : st + tile_batch_size].to(dev),
                        wh[st : st + tile_batch_size].to(dev),
                        image_size[fi].to(dev),
                        camera_params[fi].to(dev),
                        rot,
                    )
                )
                count += len(x)
        if not tokens:
            raise ValueError("Scene contains no tiles")
        ctx = self.aggregator(
            torch.cat(tokens, 0).unsqueeze(0),
            torch.ones(1, count, device=dev, dtype=torch.bool),
        )
        scene = self.scene_token(
            ctx, torch.ones(1, count, device=dev, dtype=torch.bool)
        )
        gate = torch.sigmoid(ctx[0])
        sph = torch.zeros(1, self.feature_dim, *self.pano_feature_size, device=dev)
        weight = torch.zeros(1, 1, *self.pano_feature_size, device=dev)
        cursor = 0
        for fi, tiles, xy, wh in batch_factory():
            rot = ypr_to_rot(poses[fi : fi + 1].to(dev))
            for st in range(0, tiles.shape[0], tile_batch_size):
                en = min(st + tile_batch_size, tiles.shape[0])
                x = tiles[st:en].to(dev)
                feat = self.encoder(x).unsqueeze(0)
                k = en - st
                feat = feat * gate[cursor : cursor + k].view(
                    1, 1, k, self.feature_dim, 1, 1
                )
                ps, pw = project_tile_features(
                    feat,
                    xy[st:en].to(dev).view(1, 1, k, 2),
                    wh[st:en].to(dev).view(1, 1, k, 2),
                    image_size[fi].to(dev).view(1, 1, 2),
                    camera_params[fi].to(dev).view(1, 1, 6),
                    rot.view(1, 1, 3, 3),
                    *self.pano_feature_size,
                    feature_stride=self.encoder.feature_stride
                )
                sph += ps * pw
                weight += pw
                cursor += k
        spherical = sph / weight.clamp_min(1e-6)
        spherical *= self.condition(scene).view(1, self.feature_dim, 1, 1)
        return self.decoder(spherical)
