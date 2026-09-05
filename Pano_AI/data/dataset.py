from pathlib import Path
import json
import math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

INPUT_SIZE=(224,224); PANO_SIZE=(256,512)
MIN_FRAMES=4; MAX_FRAMES=30

def resize_letterbox(image,size=INPUT_SIZE):
    image=image.convert('RGB'); th,tw=size; sw,sh=image.size
    scale=min(tw/sw,th/sh); nw,nh=max(1,round(sw*scale)),max(1,round(sh*scale))
    image=image.resize((nw,nh),Image.Resampling.BILINEAR)
    canvas=Image.new('RGB',(tw,th),(0,0,0)); canvas.paste(image,((tw-nw)//2,(th-nh)//2))
    return torch.from_numpy(np.asarray(canvas)).permute(2,0,1).float()/255

def load_panorama(path):
    image=Image.open(path).convert('RGB').resize((PANO_SIZE[1],PANO_SIZE[0]),Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image)).permute(2,0,1).float()/255

def _pinhole_from_fov(d,w,h):
    hfov=float(d['horizontal_fov_deg'])
    if not 1.0 < hfov < 179.0: raise ValueError('horizontal_fov_deg must be between 1 and 179 degrees')
    fx=w/(2.0*math.tan(math.radians(hfov)/2.0))
    vfov=2.0*math.atan((h/w)*math.tan(math.radians(hfov)/2.0))
    fy=h/(2.0*math.tan(vfov/2.0))
    return fx,fy,w/2.0,h/2.0

def load_camera_profile(scene):
    path=scene/'camera.json'
    if not path.exists():
        return {'projection':'fisheye_180','fx_norm':1.0,'fy_norm':1.0,'cx_norm':0.0,'cy_norm':0.0}
    d=json.loads(path.read_text(encoding='utf-8'))
    projection=str(d.get('projection',d.get('model','fisheye_180'))).lower().replace('-','_')
    if projection in {'fisheye','fisheye_180','equidistant','180_degree_equidistant'}:
        fov=float(d.get('fov_deg',d.get('horizontal_fov_deg',180.0)))
        if fov < 120 or fov > 200: raise ValueError(f'{path}: fisheye fov_deg should normally be 120..200')
        return {'projection':'fisheye_180','fx_norm':1.0,'fy_norm':1.0,'cx_norm':0.0,'cy_norm':0.0,'fov_deg':fov}
    if projection not in {'pinhole','perspective','phone'}:
        raise ValueError(f'{path}: unsupported projection {projection!r}; use fisheye_180 or pinhole')
    w,h=d.get('image_width'),d.get('image_height')
    if not w or not h: raise ValueError(f'{path} needs image_width and image_height')
    if all(k in d for k in ('fx','fy','cx','cy')):
        fx,fy,cx,cy=map(float,(d['fx'],d['fy'],d['cx'],d['cy']))
    elif 'horizontal_fov_deg' in d:
        fx,fy,cx,cy=_pinhole_from_fov(d,w,h)
    else:
        raise ValueError(f'{path} needs fx, fy, cx, cy or horizontal_fov_deg for pinhole projection')
    return {'projection':'pinhole','fx_norm':2.0*fx/w,'fy_norm':2.0*fy/h,'cx_norm':2.0*cx/w-1.0,'cy_norm':2.0*cy/h-1.0,'image_width':int(w),'image_height':int(h)}

class PanoramaDataset(Dataset):
    """Scene: images/* + poses.pt + panorama.png + optional camera.json.
    camera.json selects the projection per scene; N is variable (4..30).
    """
    def __init__(self,root,has_gt=True,min_frames=MIN_FRAMES,max_frames=MAX_FRAMES):
        self.root=Path(root); self.has_gt=has_gt; self.min_frames=min_frames; self.max_frames=max_frames
        if not self.root.exists(): raise FileNotFoundError(self.root)
        self.scenes=sorted(p for p in self.root.iterdir() if p.is_dir() and (p/'images').is_dir())
        if not self.scenes: raise RuntimeError(f'No scene directories found under {self.root}')
    def __len__(self): return len(self.scenes)
    def __getitem__(self,idx):
        scene=self.scenes[idx]; image_files=sorted([p for p in (scene/'images').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp'}]); n=len(image_files)
        if not n: raise RuntimeError(f'No image frames found in {scene}/images')
        if not self.min_frames <= n <= self.max_frames: raise ValueError(f'{scene}: expected {self.min_frames}..{self.max_frames} frames, got {n}')
        pose_path=scene/'poses.pt'
        if not pose_path.exists(): raise FileNotFoundError(f'Missing poses.pt in {scene}')
        poses=torch.load(pose_path,map_location='cpu',weights_only=False)
        if isinstance(poses,dict):
            for key in ('poses','ypr','rotations'):
                if key in poses: poses=poses[key]; break
        poses=torch.as_tensor(poses,dtype=torch.float32)
        if poses.ndim!=2 or poses.shape[1]!=3: raise ValueError(f'{pose_path} must contain [N,3] yaw/pitch/roll degrees')
        if n!=poses.shape[0]: raise ValueError(f'{scene}: images/poses count mismatch ({n} vs {poses.shape[0]})')
        sample={'images':torch.stack([resize_letterbox(Image.open(p)) for p in image_files]),'poses':poses,'scene':scene.name,'camera_profile':load_camera_profile(scene)}
        sample['gt_panorama']=load_panorama(scene/'panorama.png') if self.has_gt and (scene/'panorama.png').exists() else None
        if self.has_gt and sample['gt_panorama'] is None: raise FileNotFoundError(f'Missing panorama.png in {scene}')
        return sample
