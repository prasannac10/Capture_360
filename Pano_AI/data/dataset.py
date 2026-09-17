from pathlib import Path
import json, math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.high_resolution import validate_capture_size

MIN_FRAMES=4; MAX_FRAMES=30; DEFAULT_MODEL_SIZE=(224,224); DEFAULT_PANO_SIZE=(750,1500)

def resize_letterbox(image, size=DEFAULT_MODEL_SIZE):
    image=image.convert('RGB'); tw,th=size; sw,sh=image.size; scale=min(tw/sw,th/sh); nw,nh=max(1,round(sw*scale)),max(1,round(sh*scale)); image=image.resize((nw,nh),Image.Resampling.BILINEAR); canvas=Image.new('RGB',(tw,th),(0,0,0)); canvas.paste(image,((tw-nw)//2,(th-nh)//2)); return torch.from_numpy(np.asarray(canvas)).permute(2,0,1).float()/255

def load_panorama(path, size=DEFAULT_PANO_SIZE):
    image=Image.open(path).convert('RGB').resize((size[1],size[0]),Image.Resampling.BILINEAR); return torch.from_numpy(np.asarray(image)).permute(2,0,1).float()/255

def _pinhole_from_fov(d,w,h):
    hfov=float(d['horizontal_fov_deg']);
    if not 1.0<hfov<179.0: raise ValueError('horizontal_fov_deg must be between 1 and 179 degrees')
    fx=w/(2*math.tan(math.radians(hfov)/2)); vfov=2*math.atan((h/w)*math.tan(math.radians(hfov)/2)); fy=h/(2*math.tan(vfov/2)); return fx,fy,w/2,h/2

def load_camera_profile(scene):
    path=scene/'camera.json'
    if not path.exists(): raise FileNotFoundError(f'Missing {path}; high-resolution inference requires explicit camera calibration')
    d=json.loads(path.read_text(encoding='utf-8')); projection=str(d.get('projection',d.get('model','fisheye_180'))).lower().replace('-','_')
    if projection in {'fisheye','fisheye_180','equidistant','180_degree_equidistant'}:
        fov=float(d.get('fov_deg',d.get('horizontal_fov_deg',180))); w=int(d.get('image_width',9504)); h=int(d.get('image_height',6336)); validate_capture_size(w,h,'fisheye_180'); return {'projection':'fisheye_180','fx_norm':1.,'fy_norm':1.,'cx_norm':0.,'cy_norm':0.,'fov_deg':fov,'image_width':w,'image_height':h}
    if projection not in {'pinhole','perspective','phone','drone'}: raise ValueError(f'Unsupported projection {projection}')
    w,h=d.get('image_width'),d.get('image_height')
    if not w or not h: raise ValueError(f'{path} needs image_width and image_height')
    validate_capture_size(w,h,projection)
    if all(k in d for k in ('fx','fy','cx','cy')): fx,fy,cx,cy=map(float,(d['fx'],d['fy'],d['cx'],d['cy']))
    elif 'horizontal_fov_deg' in d: fx,fy,cx,cy=_pinhole_from_fov(d,w,h)
    else: raise ValueError(f'{path} needs fx/fy/cx/cy or horizontal_fov_deg')
    s=min(DEFAULT_MODEL_SIZE[1]/float(w),DEFAULT_MODEL_SIZE[0]/float(h)); nw,nh=w*s,h*s; fx2,fy2=fx*s,fy*s; cx2=cx*s+(DEFAULT_MODEL_SIZE[1]-nw)/2; cy2=cy*s+(DEFAULT_MODEL_SIZE[0]-nh)/2
    return {'projection':'pinhole','fx_norm':2*fx2/(DEFAULT_MODEL_SIZE[1]-1),'fy_norm':2*fy2/(DEFAULT_MODEL_SIZE[0]-1),'cx_norm':2*cx2/(DEFAULT_MODEL_SIZE[1]-1)-1,'cy_norm':2*cy2/(DEFAULT_MODEL_SIZE[0]-1)-1,'image_width':int(w),'image_height':int(h)}

class PanoramaDataset(Dataset):
    def __init__(self,root,has_gt=True,min_frames=MIN_FRAMES,max_frames=MAX_FRAMES,model_size=DEFAULT_MODEL_SIZE,pano_size=DEFAULT_PANO_SIZE):
        self.root=Path(root); self.has_gt=has_gt; self.min_frames=min_frames; self.max_frames=max_frames; self.model_size=model_size; self.pano_size=pano_size
        if not self.root.exists(): raise FileNotFoundError(self.root)
        self.scenes=sorted(p for p in self.root.iterdir() if p.is_dir() and (p/'images').is_dir())
        if not self.scenes: raise RuntimeError(f'No scene directories found under {self.root}')
    def __len__(self): return len(self.scenes)
    def __getitem__(self,idx):
        scene=self.scenes[idx]; files=sorted(p for p in (scene/'images').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.tif','.tiff'}); n=len(files)
        if not self.min_frames<=n<=self.max_frames: raise ValueError(f'{scene}: expected {self.min_frames}..{self.max_frames} frames, got {n}')
        poses=torch.load(scene/'poses.pt',map_location='cpu',weights_only=False) if (scene/'poses.pt').exists() else None
        if poses is None: raise FileNotFoundError(f'Missing poses.pt in {scene}')
        if isinstance(poses,dict):
            for k in ('poses','ypr','rotations'):
                if k in poses: poses=poses[k]; break
        poses=torch.as_tensor(poses,dtype=torch.float32)
        if poses.ndim!=2 or poses.shape[1]!=3 or n!=poses.shape[0]: raise ValueError(f'{scene}: poses must be [N,3] and match images')
        cp=load_camera_profile(scene); images=[]
        for p in files:
            with Image.open(p) as im:
                if (im.width,im.height)!=(cp['image_width'],cp['image_height']): raise ValueError(f'{p}: actual size {im.width}x{im.height} differs from camera.json {cp["image_width"]}x{cp["image_height"]}')
                images.append(resize_letterbox(im,self.model_size))
        sample={'images':torch.stack(images),'poses':poses,'scene':scene.name,'camera_profile':cp}; sample['gt_panorama']=load_panorama(scene/'panorama.png',self.pano_size) if self.has_gt and (scene/'panorama.png').exists() else None
        if self.has_gt and sample['gt_panorama'] is None: raise FileNotFoundError(f'Missing panorama.png in {scene}')
        return sample
