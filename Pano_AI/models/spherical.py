import math
import torch
import torch.nn.functional as F

def _equirect_dirs(height,width,device,dtype):
    theta=torch.linspace(-math.pi,math.pi,width,device=device,dtype=dtype); phi=torch.linspace(-math.pi/2,math.pi/2,height,device=device,dtype=dtype)
    phi,theta=torch.meshgrid(phi,theta,indexing='ij')
    return torch.stack((torch.cos(phi)*torch.sin(theta),torch.sin(phi),torch.cos(phi)*torch.cos(theta)),-1)

def _fisheye_grid(d):
    z=d[...,2].clamp(-1,1); theta=torch.acos(z); xy=torch.sqrt(d[...,0]**2+d[...,1]**2).clamp_min(1e-8); radius=theta/(math.pi/2)
    return torch.stack((radius*d[...,0]/xy,radius*d[...,1]/xy),-1),(theta<=math.pi/2)&(radius<=1)

def _pinhole_grid(d,k):
    fx,fy,cx,cy=[k[...,i].unsqueeze(1) for i in range(4)]; z=d[...,2].clamp_min(1e-6)
    u=fx*d[...,0]/z+cx; v=fy*d[...,1]/z+cy
    return torch.stack((u,v),-1),(d[...,2]>0)&(u>=-1)&(u<=1)&(v>=-1)&(v<=1)

def spherical_project(features,rotations,pano_h,pano_w,frame_mask=None,camera_params=None):
    """Arbitrary N views. camera_params [B,N,5]=[fx,fy,cx,cy,type], type 0=fisheye 180, 1=pinhole."""
    b,n,c,_,_=features.shape
    if camera_params is None:
        camera_params=torch.zeros(b,n,5,device=features.device,dtype=features.dtype)
    world=_equirect_dirs(pano_h,pano_w,features.device,features.dtype).unsqueeze(0).expand(b,-1,-1,-1)
    canvas=torch.zeros(b,c,pano_h,pano_w,device=features.device,dtype=features.dtype); weights=torch.zeros(b,1,pano_h,pano_w,device=features.device,dtype=features.dtype)
    for i in range(n):
        d=torch.einsum('bhwc,bcj->bhwj',world,rotations[:,i]); fg,fv=_fisheye_grid(d); pg,pv=_pinhole_grid(d,camera_params[:,i,:4]); pin=camera_params[:,i,4].view(b,1,1)>0.5
        grid=torch.where(pin.unsqueeze(-1),pg,fg); valid=torch.where(pin,pv,fv); sampled=F.grid_sample(features[:,i],grid,mode='bilinear',padding_mode='zeros',align_corners=True); vis=valid.to(features.dtype)*d[...,2].clamp_min(0)
        if frame_mask is not None: vis*=frame_mask[:,i].to(features.dtype).view(b,1,1)
        canvas+=sampled*vis.unsqueeze(1); weights+=vis.unsqueeze(1)
    return canvas/weights.clamp_min(1e-6)

def pano_to_views(pano,rotations,image_h,image_w,frame_mask=None):
    b=pano.shape[0]; device,dtype=pano.device,pano.dtype
    yy,xx=torch.meshgrid(torch.linspace(-1,1,image_h,device=device,dtype=dtype),torch.linspace(-1,1,image_w,device=device,dtype=dtype),indexing='ij'); radius=torch.sqrt(xx*xx+yy*yy); theta=radius.clamp_max(1)*(math.pi/2); xy=radius.clamp_min(1e-8)
    dirs=torch.stack((torch.sin(theta)*xx/xy,torch.sin(theta)*yy/xy,torch.cos(theta)),-1); valid=radius<=1; outputs=[]
    for i in range(rotations.shape[1]):
        world=torch.einsum('hwc,bcj->bhwj',dirs,rotations[:,i].transpose(1,2)); x=torch.atan2(world[...,0],world[...,2])/math.pi; y=world[...,1]/(math.pi/2); sampled=F.grid_sample(pano,torch.stack((x,y),-1),mode='bilinear',padding_mode='border',align_corners=True)
        if frame_mask is not None: sampled*=frame_mask[:,i].to(dtype).view(b,1,1,1)
        outputs.append(sampled)
    return torch.stack(outputs,1),valid
