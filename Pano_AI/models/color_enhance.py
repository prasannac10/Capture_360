import torch.nn as nn

class ColorEnhancementUNet(nn.Module):
    """Compact paired color/enhancement network for supervised fine-tuning."""
    def __init__(self, channels=32):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,channels,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(channels,channels,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(channels,3,3,padding=1))
    def forward(self,x): return (x+self.net(x)).clamp(0,1)
