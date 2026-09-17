"""Scene dataset that preserves original resolution through overlapping tiles."""
from pathlib import Path
import json, math
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from data.tiling import extract_tiles


def _camera(scene, width, height):
    p = scene / 'camera.json'
    d = json.loads(p.read_text()) if p.exists() else {'projection': 'fisheye_180', 'fov_deg': 180.0}
    projection = str(d.get('projection', d.get('model', 'fisheye_180'))).lower().replace('-', '_')
    if projection in {'fisheye','fisheye_180','equidistant','180_degree_equidistant'}:
        fov = float(d.get('fov_deg', d.get('horizontal_fov_deg', 180.0)))
        return torch.tensor([width/np.pi, height/np.pi, width/2, height/2, 0.0, fov], dtype=torch.float32)
    if all(k in d for k in ('fx','fy','cx','cy')):
        vals = [float(d[k]) for k in ('fx','fy','cx','cy')]
    elif 'horizontal_fov_deg' in d:
        hfov = math.radians(float(d['horizontal_fov_deg']))
        fx = width / (2*math.tan(hfov/2)); fy = fx
        vals = [fx, fy, width/2, height/2]
    else:
        raise ValueError(f'{p} needs fx/fy/cx/cy or horizontal_fov_deg')
    return torch.tensor(vals + [1.0, 0.0], dtype=torch.float32)


class VariableTilePanoramaDataset(Dataset):
    """Returns original-resolution tiles for every camera frame.

    Output per scene:
      tiles [N,T,3,tile,tile], tile_mask [N,T], tile_xy/wh [N,T,*],
      image_size [N,2], camera_params [N,6], poses [N,3].
    """
    def __init__(self, root, has_gt=True, tile_size=1024, overlap=128, min_frames=4, max_frames=30, max_tiles_per_frame=None):
        self.root=Path(root); self.has_gt=has_gt; self.tile_size=tile_size; self.overlap=overlap
        self.min_frames=min_frames; self.max_frames=max_frames; self.max_tiles_per_frame=max_tiles_per_frame
        self.scenes=sorted(p for p in self.root.iterdir() if p.is_dir() and (p/'images').is_dir())
        if not self.scenes: raise RuntimeError(f'No scenes under {self.root}')

    def __len__(self): return len(self.scenes)

    def __getitem__(self, idx):
        scene=self.scenes[idx]
        files=sorted(p for p in (scene/'images').iterdir() if p.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.tif','.tiff'})
        n=len(files)
        if not self.min_frames <= n <= self.max_frames: raise ValueError(f'{scene}: expected {self.min_frames}..{self.max_frames}, got {n}')
        poses=torch.load(scene/'poses.pt', map_location='cpu', weights_only=False)
        if isinstance(poses,dict): poses=next(poses[k] for k in ('poses','ypr','rotations') if k in poses)
        poses=torch.as_tensor(poses,dtype=torch.float32)
        if poses.shape != (n,3): raise ValueError(f'{scene}: poses must be [N,3], got {tuple(poses.shape)}')
        frame_tiles=[]; xy=[]; wh=[]; sizes=[]; cameras=[]
        for p in files:
            im=Image.open(p).convert('RGB'); w,h=im.size
            t=torch.from_numpy(np.asarray(im)).permute(2,0,1).float()/255.0
            tiles,specs=extract_tiles(t,self.tile_size,self.overlap)
            if self.max_tiles_per_frame and len(specs)>self.max_tiles_per_frame:
                raise ValueError(f'{scene}/{p.name}: {len(specs)} tiles exceeds max_tiles_per_frame={self.max_tiles_per_frame}')
            frame_tiles.append(tiles)
            xy.append(torch.tensor([[s.x,s.y] for s in specs],dtype=torch.float32))
            wh.append(torch.tensor([[s.width,s.height] for s in specs],dtype=torch.float32))
            sizes.append([w,h]); cameras.append(_camera(scene,w,h))
        tmax=max(x.shape[0] for x in frame_tiles)
        tiles=torch.zeros(n,tmax,3,self.tile_size,self.tile_size)
        tile_mask=torch.zeros(n,tmax,dtype=torch.bool); tile_xy=torch.zeros(n,tmax,2); tile_wh=torch.zeros(n,tmax,2)
        for i in range(n):
            k=frame_tiles[i].shape[0]; tiles[i,:k]=frame_tiles[i]; tile_mask[i,:k]=True; tile_xy[i,:k]=xy[i]; tile_wh[i,:k]=wh[i]
        sample={'tiles':tiles,'tile_mask':tile_mask,'tile_xy':tile_xy,'tile_wh':tile_wh,
                'image_size':torch.tensor(sizes,dtype=torch.float32),'camera_params':torch.stack(cameras),'poses':poses,'scene':scene.name}
        if self.has_gt and (scene/'panorama.png').exists():
            gt=Image.open(scene/'panorama.png').convert('RGB')
            sample['gt_panorama']=torch.from_numpy(np.asarray(gt)).permute(2,0,1).float()/255.0
        elif self.has_gt: raise FileNotFoundError(f'Missing panorama.png in {scene}')
        else: sample['gt_panorama']=None
        return sample


def tile_collate(batch):
    # Batch size 1 is recommended for 12K scenes; this collate keeps variable N/T explicit.
    if len(batch)==1: return batch[0]
    raise ValueError('VariableTilePanoramaDataset currently requires batch_size=1; variable-resolution scenes are not stackable without bucketing.')
