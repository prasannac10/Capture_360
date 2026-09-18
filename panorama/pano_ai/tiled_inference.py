"""Authoritative native-resolution tiled inference entry point."""
import argparse,json
from pathlib import Path
import yaml,torch,numpy as np
from PIL import Image
from .data.tile_dataset import VariableTilePanoramaDataset,iter_tile_batches
from .models.panorama_model import PanoramaModel
from .highres_correction import HighResolutionCorrectionPipeline
from panorama.stitching.profiles import validate_frame_set

def run_tiled_inference(scene, config_path, output_dir=None, checkpoint=None, profile="auto"):
    root=Path(__file__).resolve().parent; config_path=Path(config_path).resolve(); cfg=yaml.safe_load(config_path.read_text()); m=cfg['model']; inf=cfg['inference']; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); scene=Path(scene).resolve(); ds=VariableTilePanoramaDataset(scene.parent,False,m['tile_size'],m['tile_overlap'],cfg['input']['min_frames'],cfg['input']['max_frames']); idx=next(i for i,pth in enumerate(ds.scenes) if pth.resolve()==scene); s=ds[idx]; selected_name, selected_profile=validate_frame_set([tuple(x.int().tolist()) for x in s['image_size']],cfg['input'],profile); expected_projection=0.0 if selected_profile['projection']=='fisheye_180' else 1.0
    if not bool(torch.all(s['camera_params'][:,4]==expected_projection)):
        raise ValueError(f"{selected_name} camera.json projection does not match its configured profile")
    s['_tile_size']=m['tile_size']; s['_overlap']=m['tile_overlap']; ck=checkpoint or str((config_path.parent/inf['checkpoint']).resolve()); state=torch.load(ck,map_location=dev,weights_only=False); model=PanoramaModel(m['feature_dim'],(m['pano_feature_height'],m['pano_feature_width']),(m['output_height'],m['output_width']),m['tile_size'],m['encoder']['backbone'],m['encoder']['pretrained'],m['attention']['heads'],m['attention'].get('layers',2),inf.get('output_tile',1024)).to(dev); model.load_state_dict(state.get('model',state),strict=True); model.eval(); bs=int(inf.get('tile_batch_size',4))
    with torch.no_grad(): pred=model.forward_scene(lambda: iter_tile_batches(s,m['tile_size'],m['tile_overlap'],bs),s['image_size'],s['camera_params'],s['poses'],bs)[0]
    arr=(pred.permute(1,2,0).cpu().numpy().clip(0,1)*255).round().astype(np.uint8); out=Path(output_dir or (config_path.parent/inf['output_dir']))/s['scene']; out.mkdir(parents=True,exist_ok=True); Image.fromarray(arr).save(out/'initial_panorama.png'); Image.fromarray(arr).save(out/'initial_panorama.tiff',compression='tiff_deflate')
    corrected=HighResolutionCorrectionPipeline(cfg,dev).run(arr,None,m['tile_size'],m['tile_overlap']); Image.fromarray(corrected).save(out/'final_corrected_panorama.png'); Image.fromarray(corrected).save(out/'final_corrected_panorama.tiff',compression='tiff_deflate'); (out/'metadata.json').write_text(json.dumps({'scene':s['scene'],'input_resolution':s['image_size'].int().tolist(),'frames':len(s['frame_paths']),'tiles_per_frame':[len(x) for x in s['tile_specs']],'tile_size':m['tile_size'],'tile_overlap':m['tile_overlap'],'output_resolution':[m['output_width'],m['output_height']],'checkpoint':ck,'device':str(dev)},indent=2))
    print(out/'final_corrected_panorama.tiff')
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scene',required=True); p.add_argument('--config',default='../stitching/config.yaml'); p.add_argument('--checkpoint'); p.add_argument('--output-dir'); a=p.parse_args()
    run_tiled_inference(a.scene, Path(__file__).resolve().parent/a.config, a.output_dir, a.checkpoint)

if __name__=='__main__': main()
