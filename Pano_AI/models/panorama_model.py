import torch
import torch.nn as nn
from models.spherical import spherical_project

class PanoramaModel(nn.Module):
    """Arbitrary-N panorama model supporting fisheye and phone perspective views."""
    def __init__(self, encoder, aggregator, decoder, pano_h, pano_w):
        super().__init__(); self.encoder=encoder; self.aggregator=aggregator; self.decoder=decoder; self.pano_h=pano_h; self.pano_w=pano_w; self.last_spherical=None
    def forward(self, images, rotations, frame_mask=None, camera_params=None):
        if images.ndim != 5: raise ValueError(f'images must be [B,N,3,H,W], got {images.shape}')
        b,n=images.shape[:2]
        if frame_mask is None: frame_mask=torch.ones(b,n,device=images.device,dtype=torch.bool)
        f=self.encoder(images.reshape(b*n,*images.shape[2:])); f=f.reshape(b,n,f.shape[1],f.shape[2],f.shape[3])
        ctx=self.aggregator(f,frame_mask); pano=spherical_project(f,rotations,self.pano_h,self.pano_w,frame_mask,camera_params); self.last_spherical=pano
        return self.decoder(pano*torch.sigmoid(ctx).unsqueeze(-1).unsqueeze(-1))
