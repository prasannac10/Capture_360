"""Memory-bounded application of single-image correction models on 12K panoramas."""
from pathlib import Path
import cv2, numpy as np, torch
from .models.glare_removal import GlareRemovalUNet
from .models.color_enhance import ColorEnhancementUNet
from .models.nadir_zenith import NadirZenithInpainter
from .models.advanced_corrections import ParallaxCorrectionUNet, GhostRemovalUNet
from .models.lens_dots import remove_lens_dots
from .models.sharpen import unsharp_mask


def _tensor(x): return torch.from_numpy(x).permute(2,0,1).float().unsqueeze(0)/255.0

def _image(x): return (x[0].permute(1,2,0).detach().cpu().numpy().clip(0,1)*255).astype(np.uint8)


def apply_tiled_model(image, model, tile_size=1024, overlap=128, device='cpu', mask=None):
    h,w=image.shape[:2]; step=tile_size-overlap; out=np.zeros_like(image,dtype=np.float32); wt=np.zeros((h,w,1),np.float32)
    win=cv2.createHanningWindow((tile_size,tile_size),cv2.CV_32F); win=np.maximum(win,1e-3)[...,None]
    with torch.no_grad():
        for y in list(range(0,max(1,h-tile_size+1),step))+([max(0,h-tile_size)] if h>tile_size else []):
            for x in list(range(0,max(1,w-tile_size+1),step))+([max(0,w-tile_size)] if w>tile_size else []):
                y=min(y,h-tile_size) if h>=tile_size else 0; x=min(x,w-tile_size) if w>=tile_size else 0
                crop=image[y:min(y+tile_size,h),x:min(x+tile_size,w)]
                ph,pw=tile_size-crop.shape[0],tile_size-crop.shape[1]
                padded=cv2.copyMakeBorder(crop,0,ph,0,pw,cv2.BORDER_REFLECT_101)
                args=[_tensor(padded).to(device)]
                if mask is not None:
                    m=mask[y:min(y+tile_size,h),x:min(x+tile_size,w)]; m=cv2.copyMakeBorder(m,0,ph,0,pw,cv2.BORDER_CONSTANT,value=0)
                    args.append(torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0).to(device))
                pred=model(*args); pred=_image(pred)[:crop.shape[0],:crop.shape[1]].astype(np.float32)
                ww=win[:crop.shape[0],:crop.shape[1]]; out[y:y+crop.shape[0],x:x+crop.shape[1]]+=pred*ww; wt[y:y+crop.shape[0],x:x+crop.shape[1]]+=ww
    return np.clip(out/np.maximum(wt,1e-6),0,255).astype(np.uint8)


class HighResolutionCorrectionPipeline:
    """Apply trained correction stages without loading a full 12K tensor into every model."""
    def __init__(self, cfg, device='cpu'):
        self.device=torch.device(device); self.cfg=cfg
        t=cfg.get('correction',{}).get('toggles',{}); ck=cfg.get('correction',{}).get('checkpoints',{}); allow=cfg.get('correction',{}).get('allow_untrained',False)
        self.t=t
        self.models={}
        specs={'glare':GlareRemovalUNet,'color':ColorEnhancementUNet,'nadir_zenith':NadirZenithInpainter,'parallax':ParallaxCorrectionUNet,'ghost_removal':GhostRemovalUNet}
        for name,cls in specs.items():
            if t.get(name,False):
                m=cls().to(self.device); p=ck.get(name)
                if p and Path(p).exists(): m.load_state_dict(torch.load(p,map_location=self.device,weights_only=False).get('model',torch.load(p,map_location=self.device,weights_only=False)),strict=True)
                elif not allow: raise FileNotFoundError(f'Enabled correction checkpoint missing: {name}: {p}')
                m.eval(); self.models[name]=m
    def run(self,image,mask=None,tile_size=1024,overlap=128):
        out=image
        for name in ('glare','nadir_zenith','color','parallax','ghost_removal'):
            if name in self.models: out=apply_tiled_model(out,self.models[name],tile_size,overlap,self.device,mask if name=='nadir_zenith' else None)
        if self.t.get('dots',False): out=remove_lens_dots(out)
        if self.t.get('sharpen',False): out=unsharp_mask(out)
        return out
