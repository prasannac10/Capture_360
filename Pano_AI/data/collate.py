import torch
from models.pose import normalize_pose

def panorama_collate_fn(batch):
    max_n=max(x['images'].shape[0] for x in batch); b=len(batch); _,c,h,w=batch[0]['images'].shape
    images=torch.zeros(b,max_n,c,h,w); poses=torch.zeros(b,max_n,3); rotations=torch.eye(3).view(1,1,3,3).repeat(b,max_n,1,1); mask=torch.zeros(b,max_n,dtype=torch.bool); camera=torch.zeros(b,max_n,5)
    targets=[]
    for i,item in enumerate(batch):
        n=item['images'].shape[0]; images[i,:n]=item['images']; poses[i,:n]=item['poses']; rotations[i,:n]=normalize_pose(item['poses']); mask[i,:n]=True
        cp=item['camera_profile']; camera[i,:n,0]=cp['fx_norm']; camera[i,:n,1]=cp['fy_norm']; camera[i,:n,2]=cp['cx_norm']; camera[i,:n,3]=cp['cy_norm']; camera[i,:n,4]=1.0 if cp['projection']=='pinhole' else 0.0
        if item.get('gt_panorama') is not None: targets.append(item['gt_panorama'])
    return {'images':images,'poses':poses,'rotations':rotations,'mask':mask,'camera_params':camera,'scenes':[x['scene'] for x in batch],'gt_panorama':torch.stack(targets) if len(targets)==b else None}
