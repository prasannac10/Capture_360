"""Variable-resolution, original-resolution tiled panorama model."""
import torch
import torch.nn as nn
from models.encoder import ImageEncoder
from models.tile_metadata import TileMetadataEncoder
from models.tile_aggregator import TileAttentionAggregator, SceneToken
from models.tile_spherical import project_tile_features, ypr_to_rot
from models.decoder import MultiScalePanoramaDecoder


class PanoramaModel(nn.Module):
    """Camera/pose-aware arbitrary-N panorama model.

    Input: tiles [B,N,T,3,1024,1024], tile_mask [B,N,T], tile_xy/wh [B,N,T,2],
    image_size [B,N,2], camera_params [B,N,6], poses [B,N,3] or rotations [B,N,3,3].
    The model never resizes a complete source image to 224x224.
    """
    def __init__(self, feature_dim=64, pano_feature_size=(750,1500), output_size=(6000,12000),
                 tile_size=1024, backbone="resnet18", pretrained=True, attention_heads=8):
        super().__init__()
        self.feature_dim=feature_dim; self.pano_feature_size=pano_feature_size; self.tile_size=tile_size
        self.encoder=ImageEncoder(feature_dim, backbone, pretrained)
        self.metadata=TileMetadataEncoder(feature_dim)
        self.aggregator=TileAttentionAggregator(feature_dim, attention_heads)
        self.scene_token=SceneToken(feature_dim)
        self.condition=nn.Sequential(nn.Linear(feature_dim,feature_dim),nn.Sigmoid())
        self.decoder=MultiScalePanoramaDecoder(feature_dim,3,output_size)

    def forward(self, tiles, tile_mask, tile_xy, tile_wh, image_size, camera_params, poses=None, rotations=None):
        if tiles.ndim != 6: raise ValueError(f"tiles must be [B,N,T,3,H,W], got {tuple(tiles.shape)}")
        b,n,t,c,h,w=tiles.shape
        if rotations is None:
            if poses is None: raise ValueError("poses or rotations are required")
            rotations=torch.stack([ypr_to_rot(p) for p in poses],0)
        x=tiles.reshape(b*n*t,c,h,w)
        valid=tile_mask.reshape(-1)
        # Encode all padded slots in one shared network, then mask them from attention/projection.
        feat=self.encoder(x).reshape(b,n,t,self.feature_dim,h//8,w//8)
        xy_norm=tile_xy/torch.maximum(image_size.unsqueeze(2),torch.ones_like(image_size.unsqueeze(2)))
        wh_norm=tile_wh/torch.maximum(image_size.unsqueeze(2),torch.ones_like(image_size.unsqueeze(2)))
        im_norm=image_size.unsqueeze(2).expand(-1,-1,t,-1)
        meta=self.metadata(xy_norm,wh_norm,im_norm,camera_params.unsqueeze(2).expand(-1,-1,t,-1),rotations.unsqueeze(2).expand(-1,-1,t,-1,-1))
        tokens=feat.mean(dim=(-1,-2))+meta
        tokens=tokens.reshape(b,n*t,self.feature_dim)
        mask=tile_mask.reshape(b,n*t)
        tokens=self.aggregator(tokens,mask)
        scene=self.scene_token(tokens,mask)
        # Per-tile attention output conditions the corresponding spatial feature map.
        feat=feat.reshape(b,n*t,self.feature_dim,feat.shape[-2],feat.shape[-1])
        feat=feat*torch.sigmoid(tokens).unsqueeze(-1).unsqueeze(-1)
        feat=feat.reshape(b,n,t,self.feature_dim,feat.shape[-2],feat.shape[-1])
        spherical=project_tile_features(feat,tile_xy,image_size,camera_params,rotations,*self.pano_feature_size)
        spherical=spherical* self.condition(scene).unsqueeze(-1).unsqueeze(-1)
        return self.decoder(spherical)
