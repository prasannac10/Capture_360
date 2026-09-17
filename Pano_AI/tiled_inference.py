"""Authoritative high-resolution tiled inference entry point.

Usage from Pano_AI:
  python tiled_inference.py --scene ../data/test/scene_000001

The model consumes original-resolution tiles and decodes directly to 12000x6000.
The correction pipeline is applied tile-wise to keep memory bounded.
"""
import argparse, json
from pathlib import Path
import yaml, torch, numpy as np
from PIL import Image
from data.tile_dataset import VariableTilePanoramaDataset
from models.panorama_model import PanoramaModel
from highres_correction import HighResolutionCorrectionPipeline


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--scene',required=True); ap.add_argument('--config',default='config.yaml'); ap.add_argument('--checkpoint',default=None); ap.add_argument('--output-dir',default=None); args=ap.parse_args()
    root=Path(__file__).resolve().parent; cfg=yaml.safe_load((root/args.config).read_text()); m=cfg['model']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    scene=Path(args.scene); ds=VariableTilePanoramaDataset(scene.parent,has_gt=False,tile_size=m['tile_size'],overlap=m['tile_overlap'],min_frames=cfg['input']['min_frames'],max_frames=cfg['input']['max_frames']); idx=next(i for i,p in enumerate(ds.scenes) if p.resolve()==scene.resolve()); s=ds[idx]
    model=PanoramaModel(m['feature_dim'],(m['pano_feature_height'],m['pano_feature_width']),(m['output_height'],m['output_width']),m['tile_size'],m['encoder']['backbone'],m['encoder']['pretrained'],m['attention']['heads']).to(device)
    ck=args.checkpoint or str(root/cfg['inference']['checkpoint']); state=torch.load(ck,map_location=device,weights_only=False); model.load_state_dict(state.get('model',state),strict=True); model.eval()
    with torch.no_grad():
        pred=model(s['tiles'].unsqueeze(0).to(device),s['tile_mask'].unsqueeze(0).to(device),s['tile_xy'].unsqueeze(0).to(device),s['tile_wh'].unsqueeze(0).to(device),s['image_size'].unsqueeze(0).to(device),s['camera_params'].unsqueeze(0).to(device),poses=s['poses'].unsqueeze(0).to(device))
    initial=(pred[0].permute(1,2,0).cpu().numpy().clip(0,1)*255).astype(np.uint8)
    outdir=Path(args.output_dir or cfg['inference']['output_dir'])/s['scene']; outdir.mkdir(parents=True,exist_ok=True)
    Image.fromarray(initial).save(outdir/'initial_panorama.png'); Image.fromarray(initial).save(outdir/'initial_panorama.tiff',compression='tiff_deflate')
    corrected=HighResolutionCorrectionPipeline(cfg,device).run(initial,m.get('correction_mask'),m['tile_size'],m['tile_overlap'])
    Image.fromarray(corrected).save(outdir/'final_corrected_panorama.png'); Image.fromarray(corrected).save(outdir/'final_corrected_panorama.tiff',compression='tiff_deflate')
    meta={'scene':s['scene'],'input_resolution':s['image_size'].int().tolist(),'frames':int(s['tiles'].shape[0]),'tiles_per_frame':s['tile_mask'].sum(1).tolist(),'tile_size':m['tile_size'],'tile_overlap':m['tile_overlap'],'output_resolution':[m['output_width'],m['output_height']],'stitching':'variable-resolution tiled AI','checkpoint':ck,'device':str(device)}
    (outdir/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(outdir/'final_corrected_panorama.tiff')


if __name__=='__main__': main()
