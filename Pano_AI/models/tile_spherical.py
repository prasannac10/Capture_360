"""Projection of high-resolution tile features into a common equirectangular feature canvas."""
import math
import torch
import torch.nn.functional as F


def equirect_dirs(h,w,device,dtype):
    lon=torch.linspace(-math.pi,math.pi,w,device=device,dtype=dtype)
    lat=torch.linspace(-math.pi/2,math.pi/2,h,device=device,dtype=dtype)
    lat,lon=torch.meshgrid(lat,lon,indexing='ij')
    return torch.stack((torch.cos(lat)*torch.sin(lon),torch.sin(lat),torch.cos(lat)*torch.cos(lon)),-1)


def ypr_to_rot(ypr):
    """yaw/pitch/roll degrees -> rotation matrices [N,3,3]."""
    y,p,r=torch.deg2rad(ypr[:,0]),torch.deg2rad(ypr[:,1]),torch.deg2rad(ypr[:,2])
    z=torch.zeros_like(y); o=torch.ones_like(y)
    Rz=torch.stack((torch.cos(y),-torch.sin(y),z,z,torch.cos(y),-torch.sin(y)*0+0,z*0+0,o),-1).reshape(-1,3,3)
    # Explicit stable matrices; yaw about Y, pitch about X, roll about Z.
    cy,sy=torch.cos(y),torch.sin(y); cp,sp=torch.cos(p),torch.sin(p); cr,sr=torch.cos(r),torch.sin(r)
    Ry=torch.stack((cy,z,sy,z,o,z,-sy,z,cy),-1).reshape(-1,3,3)
    Rx=torch.stack((o,z,z,z,cp,-sp,z,sp,cp),-1).reshape(-1,3,3)
    Rz=torch.stack((cr,-sr,z,sr,cr,z,z,z,o),-1).reshape(-1,3,3)
    return Ry@Rx@Rz


def project_tile_features(tile_features, tile_xy, image_size, camera_params, rotations, pano_h, pano_w):
    """Project [N,T,C,Hf,Wf] tile features to [N,C,pano_h,pano_w].

    Tile feature pixels are mapped back to original-image coordinates before
    camera projection. This preserves variable source resolution and tile origin.
    """
    b,n,t,c,hf,wf=tile_features.shape
    device,dtype=tile_features.device,tile_features.dtype
    world=equirect_dirs(pano_h,pano_w,device,dtype).unsqueeze(0).unsqueeze(0).expand(b,n,-1,-1,-1)
    # rotations are [B,N,3,3], mapping camera rays into world coordinates.
    cam_dirs=torch.einsum('bnij,bnhwj->bnhwi',rotations.transpose(-1,-2),world)
    z=cam_dirs[...,2].clamp(-1,1)
    theta=torch.acos(z)
    xy_norm=torch.sqrt(cam_dirs[...,0]**2+cam_dirs[...,1]**2).clamp_min(1e-8)
    fov=torch.deg2rad(camera_params[...,5]).unsqueeze(-1).unsqueeze(-1).clamp_min(1e-3)
    # Equidistant fisheye radius in pixels.
    f=torch.minimum(image_size[...,0],image_size[...,1]).unsqueeze(-1).unsqueeze(-1)/(2*fov/2).clamp_min(1e-3)
    fx,fy,cx,cy=[camera_params[...,i].unsqueeze(-1).unsqueeze(-1) for i in range(4)]
    uf=fx*theta*cam_dirs[...,0]/xy_norm+cx
    vf=fy*theta*cam_dirs[...,1]/xy_norm+cy
    zsafe=cam_dirs[...,2].clamp_min(1e-6)
    up=fx*cam_dirs[...,0]/zsafe+cx; vp=fy*cam_dirs[...,1]/zsafe+cy
    pin=(camera_params[...,4]>0.5).unsqueeze(-1).unsqueeze(-1)
    u=torch.where(pin,up,uf); v=torch.where(pin,vp,vf)
    valid=torch.where(pin,(cam_dirs[...,2]>0)&(up>=0)&(up<image_size[...,0].unsqueeze(-1).unsqueeze(-1))&(vp>=0)&(vp<image_size[...,1].unsqueeze(-1).unsqueeze(-1)),
                     (theta<=fov/2)&(uf>=0)&(uf<image_size[...,0].unsqueeze(-1).unsqueeze(-1))&(vf>=0)&(vf<image_size[...,1].unsqueeze(-1).unsqueeze(-1)))
    canvas=torch.zeros(b,n,c,pano_h,pano_w,device=device,dtype=dtype); weight=torch.zeros(b,n,1,pano_h,pano_w,device=device,dtype=dtype)
    # A panorama pixel usually falls in only one/few source tiles. Iterate tiles to keep memory bounded.
    for ti in range(t):
        x0=tile_xy[:, :, ti, 0].unsqueeze(-1).unsqueeze(-1); y0=tile_xy[:, :, ti, 1].unsqueeze(-1).unsqueeze(-1)
        # feature map samples correspond to the tile's full padded coordinate extent.
        gx=(u-x0)/(image_size.new_tensor(1.0).to(device,dtype))
        gy=(v-y0)/(image_size.new_tensor(1.0).to(device,dtype))
        # Normalize using per-tile inferred size from feature stride (1024/Hf).
        tile_w=(wf*8.0); tile_h=(hf*8.0)
        gx=2*gx/(tile_w-1)-1; gy=2*gy/(tile_h-1)-1
        grid=torch.stack((gx,gy),-1).reshape(b*n,pano_h,pano_w,2)
        fti=tile_features[:,:,ti].reshape(b*n,c,hf,wf)
        sampled=F.grid_sample(fti,grid,mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,n,c,pano_h,pano_w)
        tile_valid=valid & (u>=x0)&(u<x0+tile_w)&(v>=y0)&(v<y0+tile_h)
        w=tile_valid.to(dtype).unsqueeze(2)
        canvas += sampled*w; weight += w
    out=(canvas.sum(1)/weight.sum(1).clamp_min(1e-6))
    return out
