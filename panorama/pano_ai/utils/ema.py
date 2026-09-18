import torch

class EMA:
    def __init__(self,model,decay): self.decay=decay; self.shadow={k:v.detach().clone() for k,v in model.state_dict().items()}; self.backup=None
    def update(self,model):
        for k,v in model.state_dict().items(): self.shadow[k]=self.decay*self.shadow[k]+(1-self.decay)*v.detach()
    def copy_to(self,model): self.backup={k:v.detach().clone() for k,v in model.state_dict().items()}; model.load_state_dict(self.shadow,strict=True)
    def apply_to(self,model): self.copy_to(model)
    def restore(self,model):
        if self.backup is not None: model.load_state_dict(self.backup,strict=True); self.backup=None
