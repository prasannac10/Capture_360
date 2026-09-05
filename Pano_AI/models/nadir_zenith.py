import torch
import torch.nn as nn

class NadirZenithInpainter(nn.Module):
    """Small partial-conv-style residual inpainter; fine-tune on paired pole crops."""
    def __init__(self, channels=32):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(4,channels,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(channels,channels,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(channels,3,3,padding=1))
    def forward(self,image,mask):
        return (image + self.net(torch.cat([image,mask],1))).clamp(0,1)
