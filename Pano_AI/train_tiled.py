"""Train the variable-resolution tiled panorama model.

Training uses original-resolution source tiles. For practical GPU memory use,
the target panorama is configurable (default 3000x1500); inference can decode
the learned representation to the production 12000x6000 canvas.
"""
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.tile_dataset import VariableTilePanoramaDataset, tile_collate
from models.panorama_model import PanoramaModel


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='config.yaml'); ap.add_argument('--epochs',type=int,default=None); args=ap.parse_args()
    root=Path(__file__).resolve().parent; cfg=yaml.safe_load((root/args.config).read_text())
    m=cfg['model']; tr=cfg['training']; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=VariableTilePanoramaDataset(root/tr['training_data'],True,m['tile_size'],m['tile_overlap'],cfg['input']['min_frames'],cfg['input']['max_frames'],tr.get('max_tiles_per_frame'))
    loader=DataLoader(ds,batch_size=1,shuffle=True,collate_fn=tile_collate)
    model=PanoramaModel(m['feature_dim'],(m['pano_feature_height'],m['pano_feature_width']),(m['train_output_height'],m['train_output_width']),m['tile_size'],m['encoder']['backbone'],m['encoder']['pretrained'],m['attention']['heads']).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=tr['lr'],weight_decay=tr['weight_decay']); scaler=torch.amp.GradScaler('cuda',enabled=bool(tr.get('mixed_precision',True) and device.type=='cuda'))
    epochs=args.epochs or tr['epochs']; out=root/tr['model_path']; out.mkdir(parents=True,exist_ok=True); best=float('inf')
    for epoch in range(epochs):
        model.train(); running=0.0
        for batch in loader:
            tiles=batch['tiles'].unsqueeze(0).to(device); mask=batch['tile_mask'].unsqueeze(0).to(device)
            xy=batch['tile_xy'].unsqueeze(0).to(device); wh=batch['tile_wh'].unsqueeze(0).to(device); size=batch['image_size'].unsqueeze(0).to(device)
            cam=batch['camera_params'].unsqueeze(0).to(device); poses=batch['poses'].unsqueeze(0).to(device)
            gt=batch['gt_panorama'].unsqueeze(0).to(device)
            gt=F.interpolate(gt,(m['train_output_height'],m['train_output_width']),mode='bilinear',align_corners=False)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=scaler.is_enabled()):
                pred=model(tiles,mask,xy,wh,size,cam,poses=poses)
                l1=F.l1_loss(pred,gt); loss=l1
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); running+=loss.item()
        avg=running/max(1,len(loader)); state={'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch+1,'loss':avg,'config':cfg}
        torch.save(state,out/tr['model_name'])
        if avg<best: best=avg; torch.save(state,out/tr['best_model_name'])
        print(f'epoch={epoch+1} loss={avg:.6f}')


if __name__=='__main__': main()
