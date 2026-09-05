"""Train one correction module from paired before/after samples."""
import argparse, yaml, torch
from torch.utils.data import DataLoader, random_split
from data.correction_dataset import CorrectionPairDataset
from models.glare_removal import GlareRemovalUNet
from models.nadir_zenith import NadirZenithInpainter
from models.color_enhance import ColorEnhancementUNet
from .restoration import train_one_epoch, evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['glare','nadir_zenith','color'],required=True); p.add_argument('--data',required=True); p.add_argument('--config',required=True); p.add_argument('--out',default='checkpoints'); a=p.parse_args()
    cfg=yaml.safe_load(open(a.config)); ds=CorrectionPairDataset(a.data,a.stage)
    if len(ds)<2: raise RuntimeError(f'Need paired intermediate artifacts for {a.stage}; found {len(ds)} pairs')
    n=max(1,round(.1*len(ds))); train,val=random_split(ds,[len(ds)-n,n],generator=torch.Generator().manual_seed(42))
    loader=DataLoader(train,batch_size=cfg.get('batch_size',4),shuffle=True); vloader=DataLoader(val,batch_size=cfg.get('batch_size',4))
    model={'glare':GlareRemovalUNet,'nadir_zenith':NadirZenithInpainter,'color':ColorEnhancementUNet}[a.stage](); opt=torch.optim.AdamW(model.parameters(),lr=cfg.get('lr',1e-4))
    device='cuda' if torch.cuda.is_available() else 'cpu'; model.to(device)
    best=float('inf')
    for epoch in range(cfg.get('epochs',20)):
        loss=train_one_epoch(model,loader,opt,device); m=evaluate(model,vloader,device); print(epoch+1,loss,m)
        if m['loss']<best:
            best=m['loss']; torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'metrics':m},f'{a.out}/{a.stage}_best.pt')

if __name__=='__main__': main()
