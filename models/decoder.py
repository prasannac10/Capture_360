import torch.nn as nn

class PanoramaDecoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 1)
        )

    def forward(self, x):
        return self.net(x)