import argparse, os
import torch, yaml
from torch.utils.data import DataLoader
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

def build_model(cfg, pretrained=None):
    m,e=cfg['model'],cfg['model'].get('encoder',{}); encoder=ImageEncoder(m['feature_dim'],backbone=e.get('backbone','resnet18'),pretrained=e.get('pretrained',True) if pretrained is None else pretrained); return PanoramaModel(encoder,SetAggregator(m['feature_dim']),PanoramaDecoder(m['feature_dim']),m['pano_height'],m['pano_width'])

def _loss(cfg,mode,pred,batch,device):
    if mode=='supervised': return supervised_loss(pred,batch['gt_panorama'].to(device),cfg['loss']['supervised'].get('l1_weight',1.),cfg['loss']['supervised'].get('ssim_weight',0.))
    return geometry_loss(pred,batch['images'].to(device),batch['rotations'].to(device),batch['mask'].to(device),cfg['loss']['geometry'].get('smoothness_weight',1.))

def _run_epoch(model,loader,cfg,mode,device,optimizer=None,ema=None):
    training=optimizer is not None; model.train(training); total=0.
    with torch.set_grad_enabled(training):
        for batch in loader:
            pred=model(batch['images'].to(device),batch['rotations'].to(device),batch['mask'].to(device),batch['camera_params'].to(device)); loss=_loss(cfg,mode,pred,batch,device)
            if training:
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.); optimizer.step(); ema.update(model) if ema else None
            total+=float(loss.item())
    return total/max(1,len(loader))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config.yaml'); a=p.parse_args(); cfg=yaml.safe_load(open(a.config,encoding='utf-8')); mode=cfg['training']['mode']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); inp=cfg['input']; m=cfg['model']; model_size=(inp.get('model_height',224),inp.get('model_width',224)); pano_size=(m['pano_height'],m['pano_width']); train_ds=PanoramaDataset(cfg['training']['training_data'],has_gt=mode=='supervised',model_size=model_size,pano_size=pano_size); train_loader=DataLoader(train_ds,batch_size=cfg['training']['batch_size'],shuffle=True,collate_fn=panorama_collate_fn,num_workers=0); val_loader=None; root=cfg['training'].get('val_data')
    if root and os.path.isdir(root):
        val_ds=PanoramaDataset(root,has_gt=mode=='supervised',model_size=model_size,pano_size=pano_size); val_loader=DataLoader(val_ds,batch_size=cfg['training']['batch_size'],shuffle=False,collate_fn=panorama_collate_fn,num_workers=0) if len(val_ds) else None
    model=build_model(cfg).to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training'].get('weight_decay',1e-4)); ema=EMA(model,cfg['training']['ema_decay']) if cfg['training'].get('use_ema',False) else None; os.makedirs(cfg['training']['model_path'],exist_ok=True); best=float('inf'); best_name=cfg['training'].get('best_model_name','panorama_best.pt'); last_name=cfg['training'].get('model_name','panorama_last.pt')
    for epoch in range(cfg['training']['epochs']):
        tr=_run_epoch(model,train_loader,cfg,mode,device,optimizer,ema); va=tr
        if val_loader:
            if ema: ema.copy_to(model)
            va=_run_epoch(model,val_loader,cfg,mode,device)
            if ema: ema.restore(model)
        print(f'epoch {epoch+1}/{cfg["training"]["epochs"]} | train={tr:.6f} | val={va:.6f}')
        if va<best: best=va; save_checkpoint(os.path.join(cfg['training']['model_path'],best_name),model,optimizer=optimizer,epoch=epoch+1,ema=ema)
    save_checkpoint(os.path.join(cfg['training']['model_path'],last_name),model,optimizer=optimizer,epoch=cfg['training']['epochs'],ema=ema)
if __name__=='__main__': main()
