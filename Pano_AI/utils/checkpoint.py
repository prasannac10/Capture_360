import torch
import os

def save_checkpoint(path, model):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(path, model, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
