from pathlib import Path
import cv2
import numpy as np
import torch
from models.glare_removal import GlareRemovalUNet
from models.lens_dots import remove_lens_dots
from models.nadir_zenith import NadirZenithInpainter
from models.color_enhance import ColorEnhancementUNet
from models.sharpen import unsharp_mask


def _tensor(x):
    return torch.from_numpy(x).permute(2,0,1).float().unsqueeze(0)/255.0

def _image(x):
    return (x[0].permute(1,2,0).detach().cpu().numpy().clip(0,1)*255).astype(np.uint8)

class CorrectionPipeline:
    def __init__(self, toggles=None, checkpoints=None, device='cpu'):
        self.toggles={'glare':True,'dots':True,'nadir_zenith':True,'color':True,'sharpen':True}; self.toggles.update(toggles or {})
        self.device=device; self.checkpoints=checkpoints or {}
        self.glare=GlareRemovalUNet().to(device); self.poles=NadirZenithInpainter().to(device); self.color=ColorEnhancementUNet().to(device)
        self._load(self.glare,'glare'); self._load(self.poles,'nadir_zenith'); self._load(self.color,'color')
        self.glare.eval(); self.poles.eval(); self.color.eval()
    def _load(self,model,name):
        path=self.checkpoints.get(name)
        if path and Path(path).exists():
            state=torch.load(path,map_location=self.device,weights_only=False); model.load_state_dict(state.get('model',state))
    def run(self, panorama, correction_mask=None):
        out=panorama
        with torch.no_grad():
            if self.toggles['glare']: out=_image(self.glare(_tensor(out).to(self.device)))
            if self.toggles['dots']: out=remove_lens_dots(out)
            if self.toggles['nadir_zenith'] and correction_mask is not None:
                mask=torch.from_numpy(correction_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)
                out=_image(self.poles(_tensor(out).to(self.device),mask))
            if self.toggles['color']: out=_image(self.color(_tensor(out).to(self.device)))
        if self.toggles['sharpen']: out=unsharp_mask(out)
        return out

def run_pipeline(image_paths, output_path, toggles=None, checkpoints=None):
    from stitching.opencv_stitcher import stitch_fisheye_files
    baseline=stitch_fisheye_files([Path(p) for p in image_paths], output_path)
    return CorrectionPipeline(toggles,checkpoints).run(baseline)
