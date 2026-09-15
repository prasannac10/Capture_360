from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

STAGES=('glare','dots','nadir_zenith','color')

def _tensor(path,size=None):
    im=Image.open(path).convert('RGB')
    if size: im=im.resize((size[1],size[0]),Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(im)).permute(2,0,1).float()/255

class CorrectionPairDataset(Dataset):
    """Indexes paired before/after stage images when intermediate artifacts exist.
    Expected scene layout: stages/<stage>/before.png and after.png, or before_<stage>.png/after_<stage>.png.
    """
    def __init__(self,root,stage,input_size=(256,512)):
        if stage not in STAGES: raise ValueError(stage)
        self.items=[]; root=Path(root)
        for scene in sorted(p for p in root.iterdir() if p.is_dir()):
            candidates=[(scene/'stages'/stage/'before.png',scene/'stages'/stage/'after.png'),(scene/f'before_{stage}.png',scene/f'after_{stage}.png')]
            for before,after in candidates:
                if before.exists() and after.exists(): self.items.append((before,after)); break
        self.size=input_size
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        a,b=self.items[i]; return {'input':_tensor(a,self.size),'target':_tensor(b,self.size),'stage':a.parent.name}
