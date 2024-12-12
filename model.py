import torch
from torch import nn
from network import SRCNN

class SRCNNp(nn.Module):
    def __init__(self):
        super(SRCNNp, self).__init__()
        self.SRCNN = SRCNN()
        self.SRCNN.load_state_dict(torch.load("./srcnn_x4.pth", weights_only=True, map_location = torch.device('mps')))
        self.conv = nn.Conv2d(1, 3, kernel_size = 3, stride = 1, padding = 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.SRCNN(x)
        x = self.relu(self.conv(x))
        return x