from pathlib import Path
import cv2
from models.glare_removal import GlareRemovalUNet
from models.lens_dots import remove_lens_dots
from models.nadir_zenith import NadirZenithInpainter
from models.color_enhance import ColorEnhancementUNet
from models.sharpen import unsharp_mask

class CorrectionPipeline:
    def __init__(self, toggles=None):
        self.toggles={'glare':True,'dots':True,'nadir_zenith':True,'color':True,'sharpen':True}; self.toggles.update(toggles or {})
        self.glare=GlareRemovalUNet(); self.poles=NadirZenithInpainter(); self.color=ColorEnhancementUNet()
    def run(self, panorama, correction_mask=None):
        out=panorama
        if self.toggles['glare']: out=out
        if self.toggles['dots']: out=remove_lens_dots(out)
        if self.toggles['nadir_zenith'] and correction_mask is not None:
            import torch
            with torch.no_grad(): out=self.poles(torch.from_numpy(out).permute(2,0,1).float().unsqueeze(0)/255., torch.from_numpy(correction_mask).float().unsqueeze(0).unsqueeze(0))[0].permute(1,2,0).numpy()*255
            out=out.clip(0,255).astype('uint8')
        if self.toggles['color']: out=out
        if self.toggles['sharpen']: out=unsharp_mask(out)
        return out

def run_pipeline(image_paths, output_path, toggles=None):
    from stitching.opencv_stitcher import stitch_fisheye_files
    baseline=stitch_fisheye_files([Path(p) for p in image_paths], output_path)
    return CorrectionPipeline(toggles).run(baseline)
