"""Authoritative Capture360 inference: model/OpenCV stitching + high-res correction."""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import torch
import yaml
from PIL import Image
from data.collate import panorama_collate_fn
from data.dataset import PanoramaDataset
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.encoder import ImageEncoder
from models.panorama_model import PanoramaModel
from pipeline import CorrectionPipeline
from stitching.opencv_stitcher import stitch_fisheye_files, stitch_pinhole_files
from utils.checkpoint import load_checkpoint
from utils.ema import EMA
from utils.high_resolution import OUTPUT_SIZE, resize_panorama, save_png_tiff

IMAGE_EXTENSIONS={'.png','.jpg','.jpeg','.webp','.tif','.tiff'}

def set_deterministic(seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.use_deterministic_algorithms(True,warn_only=True)

def _load_cfg(path):
    with open(path,encoding='utf-8') as f: return yaml.safe_load(f)

def _resolve(base,value):
    p=Path(value); return p if p.is_absolute() else base/p

def _build_model(cfg,device):
    m=cfg['model']; e=m.get('encoder',{}); return PanoramaModel(ImageEncoder(m['feature_dim'],e.get('backbone','resnet18'),bool(e.get('pretrained',True))),SetAggregator(m['feature_dim']),PanoramaDecoder(m['feature_dim']),m['pano_height'],m['pano_width']).to(device)

def _save_metadata(out_dir,meta): (out_dir/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')

def run_reference_inference(session,config_path,output_dir,device=None,seed=0,apply_corrections=True):
    session,config_path,output_dir=Path(session),Path(config_path),Path(output_dir); cfg=_load_cfg(config_path); set_deterministic(seed); dev=torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    dataset=PanoramaDataset(session.parent,has_gt=False,model_size=(cfg['input']['model_height'],cfg['input']['model_width']),pano_size=(cfg['model']['pano_height'],cfg['model']['pano_width'])); matches=[i for i,s in enumerate(dataset.scenes) if s.resolve()==session.resolve()]
    if not matches: raise ValueError(f'Session is not a dataset scene: {session}')
    batch=panorama_collate_fn([dataset[matches[0]]]); image_files=sorted(p for p in (session/'images').iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS); cp=batch['camera_params'][0,0]; projection='pinhole' if cp[4].item()>0.5 else 'fisheye_180'; output_size=(int(cfg['stitching'].get('output_width',OUTPUT_SIZE[0])),int(cfg['stitching'].get('output_height',OUTPUT_SIZE[1])))
    method=str(cfg['stitching'].get('method','model')).lower(); out_dir=output_dir; out_dir.mkdir(parents=True,exist_ok=True)
    if method=='opencv':
        if projection=='fisheye_180': initial_np=stitch_fisheye_files(image_files,out_dir/'opencv_initial.tiff',output_size=output_size,fov_degrees=float(cfg['stitching']['opencv'].get('fisheye_fov_deg',180.0)))
        else: initial_np=stitch_pinhole_files(image_files,output_size=output_size)
    elif method=='model':
        model=_build_model(cfg,dev); ck=_resolve(config_path.parent,cfg['inference']['checkpoint']); ema=EMA(model,cfg['training']['ema_decay']) if cfg['training'].get('use_ema',False) else None; load_checkpoint(str(ck),model,device=dev,ema=ema,use_ema=ema is not None); model.eval()
        with torch.inference_mode(): initial_t=model(batch['images'].to(dev),batch['rotations'].to(dev),batch['mask'].to(dev),batch['camera_params'].to(dev)).clamp(0,1)
        initial_np=(initial_t[0].permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8); initial_np=resize_panorama(initial_np,output_size)
    else: raise ValueError("stitching.method must be 'model' or 'opencv'")
    save_png_tiff(initial_np,out_dir/'initial_panorama')
    final_np=initial_np; correction_meta={'enabled':False,'stages':[]}
    if apply_corrections:
        c=cfg.get('correction',{}); a=cfg.get('advanced_corrections',{}); toggles={**c.get('toggles',{}),**a.get('toggles',{})}; checkpoints={k:str(_resolve(config_path.parent,v)) for k,v in {**c.get('checkpoints',{}),**a.get('checkpoints',{})}.items()}; hr=cfg.get('inference',{}).get('high_resolution',{}); correction=CorrectionPipeline(toggles,checkpoints,dev,bool(c.get('allow_untrained',False)),int(hr.get('tile_size',1024)),int(hr.get('tile_overlap',128))); mask_path=session/'correction_mask.png'; mask=np.asarray(Image.open(mask_path).convert('L'),dtype=np.float32)/255 if mask_path.exists() else None
        if mask is not None and mask.shape[:2]!=final_np.shape[:2]: mask=np.asarray(Image.fromarray((mask*255).astype(np.uint8)).resize((final_np.shape[1],final_np.shape[0]),Image.Resampling.BILINEAR),dtype=np.float32)/255
        final_np=correction.run(final_np,correction_mask=mask); correction_meta={'enabled':True,'stages':[k for k,v in toggles.items() if v]}
    save_png_tiff(final_np,out_dir/'final_corrected_panorama')
    meta={'schema_version':2,'scene':session.name,'stitching_method':method,'projection':projection,'num_frames':int(batch['mask'].sum()),'source_dimensions':[[int(dataset[matches[0]]['camera_profile']['image_width']),int(dataset[matches[0]]['camera_profile']['image_height'])]],'panorama_size':[int(final_np.shape[1]),int(final_np.shape[0])],'model_working_panorama':[int(cfg['model']['pano_width']),int(cfg['model']['pano_height'])],'corrections':correction_meta,'device':str(dev),'seed':seed,'deterministic':True,'outputs':['initial_panorama.png','initial_panorama.tiff','final_corrected_panorama.png','final_corrected_panorama.tiff','metadata.json']}; _save_metadata(out_dir,meta); return meta

def main():
    p=argparse.ArgumentParser(); p.add_argument('--session',required=True); p.add_argument('--config',default='config.yaml'); p.add_argument('--output',required=True); p.add_argument('--device',choices=['cpu','cuda'],default=None); p.add_argument('--seed',type=int,default=0); p.add_argument('--no-corrections',action='store_true'); a=p.parse_args(); run_reference_inference(a.session,a.config,a.output,a.device,a.seed,not a.no_corrections)
if __name__=='__main__': main()
