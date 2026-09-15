import math
import torch
import torch.nn.functional as F

def _equirect_dirs(h,w,device,dtype):
    t=torch.linspace(-math.pi,math.pi,w,device=device,dtype=dtype); p=torch.linspace(-math.pi/2,math.pi/2,h,device=device,dtype=dtype); p,t=torch.meshgrid(p,t,indexing='ij'); return torch.stack((torch.cos(p)*torch.sin(t),torch.sin(p),torch.cos(p)*torch.cos(t)),-1)

def _fisheye_grid(d):
    z=d[...,2].clamp(-1,1); theta=torch.acos(z); xy=torch.sqrt(d[...,0]**2+d[...,1]**2).clamp_min(1e-8); r=theta/(math.pi/2); return torch.stack((r*d[...,0]/xy,r*d[...,1]/xy),-1),(theta<=math.pi/2)&(r<=1)

def _pinhole_grid(d,k):
    fx,fy,cx,cy=[k[...,i].unsqueeze(-1).unsqueeze(-1) for i in range(4)]; z=d[...,2].clamp_min(1e-6); u=fx*d[...,0]/z+cx; v=fy*d[...,1]/z+cy; return torch.stack((u,v),-1),(d[...,2]>0)&(u>=-1)&(u<=1)&(v>=-1)&(v<=1)

def spherical_project(features,rotations,pano_h,pano_w,frame_mask=None,camera_params=None):
    """Dynamic-N projection. camera_params [B,N,5]=[fx,fy,cx,cy,type], type 0 fisheye-180 / 1 pinhole."""
    b,n,c,h,w=features.shape
    if camera_params is None: camera_params=torch.zeros(b,n,5,device=features.device,dtype=features.dtype)
    world=_equirect_dirs(pano_h,pano_w,features.device,features.dtype).unsqueeze(0).expand(b,-1,-1,-1)
    dirs=torch.einsum('bhwc,bncj->bnhwj',world,rotations)
    fg,fv=_fisheye_grid(dirs); pg,pv=_pinhole_grid(dirs,camera_params[...,:4]); pin=camera_params[...,4].unsqueeze(-1).unsqueeze(-1)>0.5
    grid=torch.where(pin.unsqueeze(-1),pg,fg); valid=torch.where(pin,pv,fv)
    sampled=F.grid_sample(features.reshape(b*n,c,h,w),grid.reshape(b*n,pano_h,pano_w,2),mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,n,c,pano_h,pano_w)
    vis=valid.to(features.dtype)*dirs[...,2].clamp_min(0)
    if frame_mask is not None: vis*=frame_mask.to(features.dtype).unsqueeze(-1).unsqueeze(-1)
    canvas=(sampled*vis.unsqueeze(2)).sum(1); weights=vis.sum(1).unsqueeze(1)
    return canvas/weights.clamp_min(1e-6)

def pano_to_views(pano,rotations,image_h,image_w,frame_mask=None):
    b=pano.shape[0]; device,dtype=pano.device,pano.dtype; yy,xx=torch.meshgrid(torch.linspace(-1,1,image_h,device=device,dtype=dtype),torch.linspace(-1,1,image_w,device=device,dtype=dtype),indexing='ij'); r=torch.sqrt(xx*xx+yy*yy); th=r.clamp_max(1)*(math.pi/2); q=r.clamp_min(1e-8); dirs=torch.stack((torch.sin(th)*xx/q,torch.sin(th)*yy/q,torch.cos(th)),-1); valid=r<=1; out=[]
    for i in range(rotations.shape[1]):
        world=torch.einsum('hwc,bcj->bhwj',dirs,rotations[:,i].transpose(1,2)); x=torch.atan2(world[...,0],world[...,2])/math.pi; y=world[...,1]/(math.pi/2); s=F.grid_sample(pano,torch.stack((x,y),-1),mode='bilinear',padding_mode='border',align_corners=True)
        if frame_mask is not None:s*=frame_mask[:,i].to(dtype).view(b,1,1,1)
        out.append(s)
    return torch.stack(out,1),valid
