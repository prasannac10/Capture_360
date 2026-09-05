import torch
import torch.nn.functional as F
from .metrics import psnr, ssim

def train_one_epoch(model, loader, optimizer, device):
    model.train(); total=0.0
    for batch in loader:
        x,y=batch['input'].to(device),batch['target'].to(device)
        pred=model(x); loss=F.l1_loss(pred,y)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); total += loss.item()
    return total/max(1,len(loader))

def evaluate(model, loader, device):
    model.eval(); losses=[]; ps=[]; ss=[]
    with torch.no_grad():
        for batch in loader:
            x,y=batch['input'].to(device),batch['target'].to(device); p=model(x); losses.append(F.l1_loss(p,y).item()); ps.append(psnr(p,y)); ss.append(ssim(p,y))
    return {'loss':sum(losses)/max(1,len(losses)),'psnr':sum(ps)/max(1,len(ps)),'ssim':sum(ss)/max(1,len(ss))}
