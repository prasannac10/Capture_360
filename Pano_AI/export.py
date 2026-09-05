import argparse,json,os,torch,yaml
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.encoder import ImageEncoder
from models.panorama_model import PanoramaModel
from utils.checkpoint import load_checkpoint
from utils.ema import EMA

INPUT_H=224; INPUT_W=224; DEFAULT_N=6

def build_model(cfg):
    return PanoramaModel(ImageEncoder(cfg['model']['feature_dim']),SetAggregator(cfg['model']['feature_dim']),PanoramaDecoder(cfg['model']['feature_dim']),cfg['model']['pano_height'],cfg['model']['pano_width'])

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config.yaml'); p.add_argument('--checkpoint',default=None); p.add_argument('--output',default='artifacts/pano_model.onnx'); p.add_argument('--num-frames',type=int,default=DEFAULT_N); a=p.parse_args()
    with open(a.config,encoding='utf-8') as f: cfg=yaml.safe_load(f)
    model=build_model(cfg).eval(); ema=EMA(model,cfg['training']['ema_decay']) if cfg['training'].get('use_ema',False) else None
    checkpoint=a.checkpoint or os.path.join(cfg['training']['model_path'],cfg['training']['model_name']); load_checkpoint(checkpoint,model,device='cpu',ema=ema,use_ema=ema is not None)
    n=a.num_frames; images=torch.zeros(1,n,3,INPUT_H,INPUT_W); rotations=torch.eye(3).view(1,1,3,3).repeat(1,n,1,1); camera_params=torch.zeros(1,n,5); camera_params[...,4]=0; mask=torch.ones(1,n,dtype=torch.bool)
    os.makedirs(os.path.dirname(a.output) or '.',exist_ok=True)
    torch.onnx.export(model,(images,rotations,mask,camera_params),a.output,input_names=['images','rotations','frame_mask','camera_params'],output_names=['panorama'],dynamic_axes={'images':{1:'num_frames'},'rotations':{1:'num_frames'},'frame_mask':{1:'num_frames'},'camera_params':{1:'num_frames'}},opset_version=20,dynamo=False)
    meta={'runtime':'onnxruntime-android','dynamic_num_frames':True,'supported_num_frames':[4,30],'recommended_fisheye_num_frames':[4,8],'recommended_phone_num_frames':[20,30],'input_images':[1,'N',3,224,224],'input_rotations':[1,'N',3,3],'input_frame_mask':[1,'N'],'input_camera_params':[1,'N',5],'camera_params':'[fx_norm,fy_norm,cx_norm,cy_norm,projection_type]; projection_type 0=fisheye-180, 1=pinhole','input_normalization':'RGB / 255; letterbox 224x224','output_panorama':[1,3,cfg['model']['pano_height'],cfg['model']['pano_width']]}
    with open(os.path.join(os.path.dirname(a.output),'model_metadata.json'),'w',encoding='utf-8') as f: json.dump(meta,f,indent=2)
    print('Exported',a.output)
if __name__=='__main__': main()
