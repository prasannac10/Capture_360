import math
import torch

def psnr(pred,target):
    mse=torch.mean((pred-target)**2).item(); return 99.0 if mse==0 else 10*math.log10(1.0/mse)

def ssim(pred,target):
    # Lightweight global SSIM proxy; use torchmetrics/skimage for full SSIM in production.
    mu_x,mu_y=pred.mean(),target.mean(); vx,vy=pred.var(),target.var(); c1=.01**2; c2=.03**2
    cov=((pred-mu_x)*(target-mu_y)).mean()
    return float(((2*mu_x*mu_y+c1)*(2*cov+c2))/((mu_x**2+mu_y**2+c1)*(vx+vy+c2)))

def lpips(pred,target):
    try:
        import lpips
        net=lpips.LPIPS(net='alex').to(pred.device)
        return float(net(pred*2-1,target*2-1).mean().detach().cpu())
    except ImportError:
        return None
