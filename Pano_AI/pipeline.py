"""Deterministic high-resolution panorama correction pipeline."""
from pathlib import Path
import warnings
import cv2
import numpy as np
import torch
from models.color_enhance import ColorEnhancementUNet
from models.glare_removal import GlareRemovalUNet
from models.lens_dots import remove_lens_dots
from models.nadir_zenith import NadirZenithInpainter
from models.sharpen import unsharp_mask
from models.advanced_corrections import ParallaxCorrectionUNet, GhostRemovalUNet, SeamBlendingUNet, OverlapDetectionUNet, ParallaxCorrectionDetector
from utils.high_resolution import apply_tiled

DEFAULT_TOGGLES={'glare':True,'dots':True,'nadir_zenith':True,'color':True,'sharpen':True,'parallax':False,'ghost_removal':False,'seam_blending':False,'overlap_detection':False,'parallax_flow':False}

def _tensor(x): return torch.from_numpy(x).permute(2,0,1).float().unsqueeze(0)/255.
def _image(x): return (x[0].permute(1,2,0).detach().cpu().numpy().clip(0,1)*255).round().astype(np.uint8)

def _load_checkpoint(model,path,name,device,allow_untrained=False):
    if not path:
        if allow_untrained: warnings.warn(f'No checkpoint configured for {name}; using untrained weights.'); return False
        raise FileNotFoundError(f"Correction stage '{name}' is enabled but no checkpoint was configured")
    path=Path(path)
    if not path.exists():
        if allow_untrained: warnings.warn(f'Checkpoint not found for {name}: {path}; using untrained weights.'); return False
        raise FileNotFoundError(f"Checkpoint not found for enabled correction stage '{name}': {path}")
    state=torch.load(path,map_location=device,weights_only=False); model.load_state_dict(state.get('model',state),strict=True); return True

class CorrectionPipeline:
    """Apply learned correction stages to a 12Kx6K panorama using bounded-memory tiles."""
    def __init__(self,toggles=None,checkpoints=None,device='cpu',allow_untrained=False,tile_size=1024,tile_overlap=128):
        self.toggles=DEFAULT_TOGGLES.copy(); self.toggles.update(toggles or {}); self.device=torch.device(device); self.checkpoints=checkpoints or {}; self.allow_untrained=allow_untrained; self.tile_size=tile_size; self.tile_overlap=tile_overlap
        self.glare=GlareRemovalUNet().to(self.device); self.poles=NadirZenithInpainter().to(self.device); self.color=ColorEnhancementUNet().to(self.device); self.parallax=ParallaxCorrectionUNet().to(self.device); self.ghost_removal=GhostRemovalUNet().to(self.device); self.seam_blending=SeamBlendingUNet().to(self.device); self.overlap_detection=OverlapDetectionUNet().to(self.device); self.parallax_flow=ParallaxCorrectionDetector().to(self.device)
        for name,model in [('glare',self.glare),('nadir_zenith',self.poles),('color',self.color),('parallax',self.parallax),('ghost_removal',self.ghost_removal),('seam_blending',self.seam_blending),('overlap_detection',self.overlap_detection),('parallax_flow',self.parallax_flow)]:
            if self.toggles.get(name,False): _load_checkpoint(model,self.checkpoints.get(name),name,self.device,allow_untrained)
            model.eval()

    def _apply_model(self,model,image,mask=None):
        def fn(tile,tile_mask):
            with torch.inference_mode():
                x=_tensor(tile).to(self.device)
                if tile_mask is not None: return _image(model(x,torch.from_numpy(tile_mask).float().unsqueeze(0).unsqueeze(0).to(self.device)))
                return _image(model(x))
        return apply_tiled(image,fn,self.tile_size,self.tile_overlap,mask)

    def run(self,panorama,correction_mask=None,auxiliary_data=None):
        out=panorama.copy(); auxiliary_data=auxiliary_data or {}
        if self.toggles['glare']: out=self._apply_model(self.glare,out)
        if self.toggles['dots']: out=remove_lens_dots(out)
        if self.toggles['nadir_zenith'] and correction_mask is not None: out=self._apply_model(self.poles,out,correction_mask)
        if self.toggles['color']: out=self._apply_model(self.color,out)
        if self.toggles['parallax']: out=self._apply_model(self.parallax,out)
        if self.toggles['parallax_flow']:
            ref=auxiliary_data.get('parallax_reference')
            if ref is not None:
                with torch.inference_mode(): self.last_parallax_flow=self.parallax_flow(_tensor(out).to(self.device),_tensor(ref).to(self.device),None)
        if self.toggles['ghost_removal']:
            mask=auxiliary_data.get('ghost_mask'); out=self._apply_model(self.ghost_removal,out,mask)
        if self.toggles['seam_blending']:
            ref=auxiliary_data.get('seam_reference'); edges=auxiliary_data.get('seam_edges')
            if ref is not None and edges is not None:
                def blend(tile,tile_edges):
                    y,x=tile.shape[:2]; ref_tile=ref[:y,:x] if ref.shape[:2]==tile.shape[:2] else cv2.resize(ref,(x,y),interpolation=cv2.INTER_LINEAR); edge=cv2.resize(edges,(x,y),interpolation=cv2.INTER_LINEAR) if edges.shape[:2]!=tile.shape[:2] else edges
                    with torch.inference_mode():
                        w=self.seam_blending(_tensor(tile).to(self.device),_tensor(ref_tile).to(self.device),torch.from_numpy(edge).float().unsqueeze(0).unsqueeze(0).to(self.device)); return _image(_tensor(tile).to(self.device)*w+_tensor(ref_tile).to(self.device)*(1-w))
                out=apply_tiled(out,blend,self.tile_size,self.tile_overlap,edges)
        if self.toggles['overlap_detection']:
            ref=auxiliary_data.get('overlap_reference')
            if ref is not None:
                with torch.inference_mode(): self.last_overlap_results=self.overlap_detection(_tensor(out).to(self.device),_tensor(ref).to(self.device))
        if self.toggles['sharpen']: out=unsharp_mask(out)
        return out

def run_pipeline(image_paths,output_path,toggles=None,checkpoints=None,device='cpu',allow_untrained=False,output_size=(12000,6000),fov_degrees=180.0):
    from stitching.opencv_stitcher import stitch_fisheye_files
    baseline=stitch_fisheye_files([Path(p) for p in image_paths],output_path,output_size=output_size,fov_degrees=fov_degrees)
    return CorrectionPipeline(toggles,checkpoints,device,allow_untrained).run(baseline)
