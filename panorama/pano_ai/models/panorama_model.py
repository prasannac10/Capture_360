"""Variable-resolution panorama model with re-iterable tile streaming."""

import torch
import torch.nn as nn
from .encoder import ImageEncoder
from .tile_metadata import TileMetadataEncoder
from .tile_aggregator import TileAttentionAggregator, SceneToken
from .tile_spherical import project_tile_features, normalize_camera_features, ypr_to_rot
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
        return self.metadata(xy, wh, image, c, r)[0]

    def forward_scene(
        self, batch_factory, image_size, camera_params, poses, tile_batch_size=4
    ):
        """Yield (frame_index, tiles[K,3,H,W], xy[K,2], wh[K,2]) twice.

        All chunks of each frame must be contiguous, as in iter_tile_batches.
        """
        dev = next(self.parameters()).device
        if tile_batch_size <= 0:
            raise ValueError('tile_batch_size must be positive')
        rotations = poses.to(dev)
        if rotations.ndim == 2:
            rotations = ypr_to_rot(rotations)
        if rotations.shape != (len(image_size), 3, 3):
            raise ValueError('poses must be [N,3] angles or [N,3,3] camera-to-world rotations')
        tokens = []
        count = 0
        for fi, tiles, xy, wh in batch_factory():
            for st in range(0, tiles.shape[0], tile_batch_size):
                x = tiles[st : st + tile_batch_size].to(dev)
                rot = rotations[fi : fi + 1]
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
        current_frame = None
        finished_frames = set()
        frame_sum = frame_weight = camera_weight = None

        def blend_frame(sph, weight, frame_sum, frame_weight, camera_weight):
            features, confidence = normalize_camera_features(frame_sum, frame_weight, camera_weight)
            return sph + features * confidence, weight + confidence

        for fi, tiles, xy, wh in batch_factory():
            if fi != current_frame:
                if current_frame is not None:
                    sph, weight = blend_frame(sph, weight, frame_sum, frame_weight, camera_weight)
                    finished_frames.add(current_frame)
                if fi in finished_frames:
                    raise ValueError('Tile factory must keep each frame contiguous')
                current_frame = fi
                frame_sum = torch.zeros_like(sph)
                frame_weight = torch.zeros_like(weight)
            rot = rotations[fi : fi + 1]
            for st in range(0, tiles.shape[0], tile_batch_size):
                en = min(st + tile_batch_size, tiles.shape[0])
                x = tiles[st:en].to(dev)
                feat = self.encoder(x).unsqueeze(0).unsqueeze(0)
                k = en - st
                feat = feat * gate[cursor : cursor + k].view(
                    1, 1, k, self.feature_dim, 1, 1
                )
                ps, pw, cw = project_tile_features(
                    feat,
                    xy[st:en].to(dev).view(1, 1, k, 2),
                    wh[st:en].to(dev).view(1, 1, k, 2),
                    image_size[fi].to(dev).view(1, 1, 2),
                    camera_params[fi].to(dev).view(1, 1, 6),
                    rot.view(1, 1, 3, 3),
                    *self.pano_feature_size,
                    feature_stride=self.encoder.feature_stride,
                    return_per_camera=True,
                )
                frame_sum = frame_sum + ps[:, 0]
                frame_weight = frame_weight + pw[:, 0]
                camera_weight = cw[:, 0]
                cursor += k
        if cursor != count:
            raise ValueError('Tile factory must yield the same tiles on both passes')
        sph, weight = blend_frame(sph, weight, frame_sum, frame_weight, camera_weight)
        spherical = sph / weight.clamp_min(1e-12)
        spherical = spherical * self.condition(scene).view(1, self.feature_dim, 1, 1)
        return self.decoder(spherical)
