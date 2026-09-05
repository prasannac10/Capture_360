from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

INPUT_SIZE=(224,224); PANO_SIZE=(256,512)

def resize_letterbox(image,size=INPUT_SIZE):
    image=image.convert('RGB'); th,tw=size; sw,sh=image.size; scale=min(tw/sw,th/sh); nw,nh=max(1,round(sw*scale)),max(1,round(sh*scale)); image=image.resize((nw,nh),Image.Resampling.BILINEAR); canvas=Image.new('RGB',(tw,th),(0,0,0)); canvas.paste(image,((tw-nw)//2,(th-nh)//2)); return torch.from_numpy(np.asarray(canvas)).permute(2,0,1).float()/255

def load_panorama(path):
    image=Image.open(path).convert('RGB').resize((PANO_SIZE[1],PANO_SIZE[0]),Image.Resampling.BILINEAR); return torch.from_numpy(np.asarray(image)).permute(2,0,1).float()/255

def load_camera_profile(scene):
    path=scene/'camera.json'
    if not path.exists(): return {'projection':'fisheye_180','fx_norm':1.0,'fy_norm':1.0,'cx_norm':0.0,'cy_norm':0.0}
    d=json.loads(path.read_text(encoding='utf-8')); projection=d.get('projection','fisheye_180').lower()
    if projection.startswith('fish'): return {'projection':'fisheye_180','fx_norm':1.0,'fy_norm':1.0,'cx_norm':0.0,'cy_norm':0.0}
    if not all(k in d for k in ('fx','fy','cx','cy')): raise ValueError(f'{path} needs fx, fy, cx, cy for pinhole projection')
    # Normalize OpenCV pixel intrinsics to ONNX grid coordinates [-1,1].
    w,h=d.get('image_width'),d.get('image_height')
    if not w or not h: raise ValueError(f'{path} needs image_width and image_height')
    return {'projection':'pinhole','fx_norm':2*d['fx']/w,'fy_norm':2*d['fy']/h,'cx_norm':2*d['cx']/w-1,'cy_norm':2*d['cy']/h-1}

class PanoramaDataset(Dataset):
    """Scene: images/*.png + poses.pt + optional panorama.png + optional camera.json."""
    def __init__(self,root,has_gt=True):
        self.root=Path(root); self.has_gt=has_gt
        if not self.root.exists(): raise FileNotFoundError(self.root)
        self.scenes=sorted(p for p in self.root.iterdir() if p.is_dir() and (p/'images').is_dir())
        if not self.scenes: raise RuntimeError(f'No scene directories found under {self.root}')
    def __len__(self): return len(self.scenes)
    def __getitem__(self,idx):
        scene=self.scenes[idx]; image_files=sorted((scene/'images').glob('*.png'))
        if not image_files: raise RuntimeError(f'No PNG frames found in {scene}/images')
        pose_path=scene/'poses.pt'
        if not pose_path.exists(): raise FileNotFoundError(f'Missing poses.pt in {scene}')
        poses=torch.load(pose_path,map_location='cpu',weights_only=False)
        if isinstance(poses,dict):
            for key in ('poses','ypr','rotations'):
                if key in poses: poses=poses[key]; break
        poses=torch.as_tensor(poses,dtype=torch.float32)
        if poses.ndim!=2 or poses.shape[1]!=3: raise ValueError(f'{pose_path} must contain [N,3] yaw/pitch/roll degrees')
        if len(image_files)!=poses.shape[0]: raise ValueError(f'{scene}: images/poses count mismatch')
        sample={'images':torch.stack([resize_letterbox(Image.open(p)) for p in image_files]),'poses':poses,'scene':scene.name,'camera_profile':load_camera_profile(scene)}
        sample['gt_panorama']=load_panorama(scene/'panorama.png') if self.has_gt and (scene/'panorama.png').exists() else None
        if self.has_gt and sample['gt_panorama'] is None: raise FileNotFoundError(f'Missing panorama.png in {scene}')
        return sample
