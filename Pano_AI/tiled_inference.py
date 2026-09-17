"""Authoritative native-resolution tiled inference entry point."""
import argparse,json
from pathlib import Path
import yaml,torch,numpy as np
from PIL import Image
from data.tile_dataset import VariableTilePanoramaDataset,iter_tile_batches
from models.panorama_model import PanoramaModel
from highres_correction import HighResolutionCorrectionPipeline

def main():
    p=argparse.ArgumentParser(); p.add_argument('--scene',required=True); p.add_argument('--config',default='config.yaml'); p.add_argument('--checkpoint'); p.add_argument('--output-dir'); a=p.parse_args(); root=Path(__file__).resolve().parent; cfg=yaml.safe_load((root/a.config).read_text()); m=cfg['model']; inf=cfg['inference']; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); scene=Path(a.scene).resolve(); ds=VariableTilePanoramaDataset(scene.parent,False,m['tile_size'],m['tile_overlap'],cfg['input']['min_frames'],cfg['input']['max_frames']); idx=next(i for i,pth in enumerate(ds.scenes) if pth.resolve()==scene); s=ds[idx]; s['_tile_size']=m['tile_size']; s['_overlap']=m['tile_overlap']; ck=a.checkpoint or str(root/inf['checkpoint']); state=torch.load(ck,map_location=dev,weights_only=False); model=PanoramaModel(m['feature_dim'],(m['pano_feature_height'],m['pano_feature_width']),(m['output_height'],m['output_width']),m['tile_size'],m['encoder']['backbone'],m['encoder']['pretrained'],m['attention']['heads'],m['attention'].get('layers',2),inf.get('output_tile',1024)).to(dev); model.load_state_dict(state.get('model',state),strict=True); model.eval(); bs=int(inf.get('tile_batch_size',4))
    with torch.no_grad(): pred=model.forward_scene(lambda: iter_tile_batches(s,m['tile_size'],m['tile_overlap'],bs),s['image_size'],s['camera_params'],s['poses'],bs)[0]
    arr=(pred.permute(1,2,0).cpu().numpy().clip(0,1)*255).round().astype(np.uint8); out=Path(a.output_dir or root/inf['output_dir'])/s['scene']; out.mkdir(parents=True,exist_ok=True); Image.fromarray(arr).save(out/'initial_panorama.png'); Image.fromarray(arr).save(out/'initial_panorama.tiff',compression='tiff_deflate')
    corrected=HighResolutionCorrectionPipeline(cfg,dev).run(arr,None,m['tile_size'],m['tile_overlap']); Image.fromarray(corrected).save(out/'final_corrected_panorama.png'); Image.fromarray(corrected).save(out/'final_corrected_panorama.tiff',compression='tiff_deflate'); (out/'metadata.json').write_text(json.dumps({'scene':s['scene'],'input_resolution':s['image_size'].int().tolist(),'frames':len(s['frame_paths']),'tiles_per_frame':[len(x) for x in s['tile_specs']],'tile_size':m['tile_size'],'tile_overlap':m['tile_overlap'],'output_resolution':[m['output_width'],m['output_height']],'checkpoint':ck,'device':str(dev)},indent=2))
    print(out/'final_corrected_panorama.tiff')
if __name__=='__main__': main()
