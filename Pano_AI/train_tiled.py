"""Native-resolution tiled panorama training with validation and EMA."""
import argparse
from pathlib import Path
import yaml,torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.tile_dataset import VariableTilePanoramaDataset,tile_collate,iter_tile_batches
from models.panorama_model import PanoramaModel
from utils.ema import EMA

def run_scene(model,sample,device,tile_batch_size,train,optimizer=None,scaler=None):
    def factory(): return iter_tile_batches(sample,model.encoder.feature_stride*128,model.encoder.feature_stride*16,tile_batch_size)
    # tile_size/overlap are supplied by caller through model attributes in the closure below.
    if train: optimizer.zero_grad(set_to_none=True)
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        pred=model.forward_scene(lambda: iter_tile_batches(sample,sample['_tile_size'],sample['_overlap'],tile_batch_size),sample['image_size'],sample['camera_params'],sample['poses'],tile_batch_size)
        gt=sample['gt_panorama'].unsqueeze(0).to(device); gt=F.interpolate(gt,(pred.shape[-2],pred.shape[-1]),mode='bilinear',align_corners=False); loss=F.l1_loss(pred,gt)
        if train:
            if scaler and scaler.is_enabled(): scaler.scale(loss).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(optimizer); scaler.update()
            else: loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); optimizer.step()
    return float(loss.detach().item())

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='config.yaml'); p.add_argument('--epochs',type=int); a=p.parse_args(); root=Path(__file__).resolve().parent; cfg=yaml.safe_load((root/a.config).read_text()); m=cfg['model']; tr=cfg['training']; dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tile_bs=int(tr.get('tile_batch_size',cfg.get('inference',{}).get('tile_batch_size',4)))
    train=VariableTilePanoramaDataset(root/tr['training_data'],True,m['tile_size'],m['tile_overlap'],cfg['input']['min_frames'],cfg['input']['max_frames'],tr.get('max_tiles_per_frame')); val=VariableTilePanoramaDataset(root/tr['val_data'],True,m['tile_size'],m['tile_overlap'],cfg['input']['min_frames'],cfg['input']['max_frames'],tr.get('max_tiles_per_frame')) if tr.get('val_data') and (root/tr['val_data']).exists() else None
    model=PanoramaModel(m['feature_dim'],(m['pano_feature_height'],m['pano_feature_width']),(m['train_output_height'],m['train_output_width']),m['tile_size'],m['encoder']['backbone'],m['encoder']['pretrained'],m['attention']['heads'],m['attention'].get('layers',2),tr.get('decoder_output_tile',1024)).to(dev); opt=torch.optim.AdamW(model.parameters(),lr=tr['lr'],weight_decay=tr['weight_decay']); scaler=torch.amp.GradScaler('cuda',enabled=bool(tr.get('mixed_precision',True) and dev.type=='cuda')); ema=EMA(model,tr['ema_decay']) if tr.get('use_ema',False) else None; out=root/tr['model_path']; out.mkdir(parents=True,exist_ok=True); best=float('inf'); epochs=a.epochs or tr['epochs']
    def prep(s): s=dict(s); s['_tile_size']=m['tile_size']; s['_overlap']=m['tile_overlap']; return s
    for ep in range(1,epochs+1):
        model.train(); tl=sum(run_scene(model,prep(train[i]),dev,tile_bs,True,opt,scaler) for i in torch.randperm(len(train)).tolist())/max(1,len(train));
        if ema: ema.update(model); ema.copy_to(model)
        model.eval(); vl=sum(run_scene(model,prep(val[i]),dev,tile_bs,False) for i in range(len(val)))/max(1,len(val)) if val else tl
        if ema: ema.restore(model)
        state={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':ep,'train_loss':tl,'val_loss':vl,'ema':ema.shadow if ema else None,'config':cfg}; torch.save(state,out/tr['model_name']);
        if vl<best: best=vl; torch.save(state,out/tr['best_model_name'])
        print(f'epoch={ep}/{epochs} train={tl:.6f} val={vl:.6f}')
if __name__=='__main__': main()
