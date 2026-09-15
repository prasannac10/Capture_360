import os, torch

def save_checkpoint(path,model,optimizer=None,epoch=None,ema=None):
    d=os.path.dirname(path)
    if d: os.makedirs(d,exist_ok=True)
    x={'model':model.state_dict()}
    if optimizer is not None:x['optimizer']=optimizer.state_dict()
    if epoch is not None:x['epoch']=epoch
    if ema is not None:x['ema']=ema.shadow
    torch.save(x,path)

def load_checkpoint(path,model,device='cpu',ema=None,use_ema=False,optimizer=None):
    x=torch.load(path,map_location=device,weights_only=False); model.load_state_dict(x.get('model',x),strict=True)
    if optimizer is not None and 'optimizer' in x: optimizer.load_state_dict(x['optimizer'])
    if ema is not None and 'ema' in x: ema.shadow={k:v.to(device) for k,v in x['ema'].items()}; ema.copy_to(model) if use_ema else None
    return x
