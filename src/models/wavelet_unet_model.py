# src/models/wavelet_unet_model.py
import torch
import torch.nn as nn
from src.models.wavelet_ops import wt, iwt


class Waveletnet(nn.Module):
    def __init__(self):
        super().__init__()
        c = 16

        self.conv1  = nn.Conv2d(12,   c,    3, 1, padding=1)
        self.conv2  = nn.Conv2d(4*c,  4*c,  3, 1, padding=1)
        self.conv3  = nn.Conv2d(16*c, 16*c, 3, 1, padding=1)
        self.conv4  = nn.Conv2d(64*c, 64*c, 3, 1, padding=1)

        self.convd1 = nn.Conv2d(c,     12,  3, 1, padding=1)
        self.convd2 = nn.Conv2d(2*c,   c,   3, 1, padding=1)
        self.convd3 = nn.Conv2d(8*c,   4*c, 3, 1, padding=1)
        self.convd4 = nn.Conv2d(32*c,  16*c,3, 1, padding=1)

        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        w1 = wt(x)
        c1 = self.relu(self.conv1(w1))

        w2 = wt(c1)
        c2 = self.relu(self.conv2(w2))

        w3 = wt(c2)
        c3 = self.relu(self.conv3(w3))

        w4 = wt(c3)
        c4 = self.relu(self.conv4(w4))

        c5 = self.relu(self.conv4(c4))
        c6 = self.conv4(c5)

        ic4 = self.relu(c6 + w4)
        iw4 = iwt(ic4)
        iw4 = torch.cat([c3, iw4], dim=1)

        ic3 = self.relu(self.convd4(iw4))
        iw3 = iwt(ic3)
        iw3 = torch.cat([c2, iw3], dim=1)

        ic2 = self.relu(self.convd3(iw3))
        iw2 = iwt(ic2)
        iw2 = torch.cat([c1, iw2], dim=1)

        ic1 = self.relu(self.convd2(iw2))
        iw1 = self.relu(self.convd1(ic1))

        y = iwt(iw1)
        return y


class ACT(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = Waveletnet()
        self.c = nn.Conv2d(3, 3, 1, padding=0, bias=False)

    def forward(self, x):
        y = self.net(x)
        y1 = self.c(y)
        return y + y1