import argparse,os,torch,yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.collate import panorama_collate_fn
from data.dataset import PanoramaDataset
from losses.geometry import geometry_loss
from losses.supervised import supervised_loss
from models.aggregator import SetAggregator
from models.decoder import PanoramaDecoder
from models.encoder import ImageEncoder
from models.panorama_model import PanoramaModel
from utils.checkpoint import save_checkpoint
from utils.ema import EMA

def build_model(cfg): return PanoramaModel(ImageEncoder(cfg['model']['feature_dim']),SetAggregator(cfg['model']['feature_dim']),PanoramaDecoder(cfg['model']['feature_dim']),cfg['model']['pano_height'],cfg['model']['pano_width'])
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='config.yaml');a=p.parse_args();cfg=yaml.safe_load(open(a.config,encoding='utf-8'));mode=cfg['training']['mode'];device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 ds=PanoramaDataset(cfg['training']['training_data'],has_gt=mode=='supervised'); dl=DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=True,collate_fn=panorama_collate_fn,num_workers=0); model=build_model(cfg).to(device);opt=torch.optim.Adam(model.parameters(),lr=cfg['training']['lr']);ema=EMA(model,cfg['training']['ema_decay']) if cfg['training'].get('use_ema',False) else None
 for epoch in range(cfg['training']['epochs']):
  model.train();running=0
  for batch in tqdm(dl,desc=f'epoch {epoch+1}/{cfg["training"]["epochs"]}'):
   images,rot,mask,camera=[batch[k].to(device) for k in ('images','rotations','mask','camera_params')];pred=model(images,rot,mask,camera)
   loss=supervised_loss(pred,batch['gt_panorama'].to(device),cfg['loss']['supervised'].get('l1_weight',1),cfg['loss']['supervised'].get('ssim_weight',0)) if mode=='supervised' else geometry_loss(pred,images,rot,mask,cfg['loss']['geometry'].get('smoothness_weight',1))
   opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();ema.update(model) if ema else None;running+=loss.item()
  print(f'epoch {epoch+1} | loss {running/max(1,len(dl)):.6f}')
 ck=os.path.join(cfg['training']['model_path'],cfg['training']['model_name']);save_checkpoint(ck,model,optimizer=opt,epoch=cfg['training']['epochs'],ema=ema);print('saved checkpoint:',ck)
if __name__=='__main__':main()
