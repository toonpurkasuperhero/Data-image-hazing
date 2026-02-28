# src/models/unetpp_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNetPP(nn.Module):
    """
    A compact U-Net++ (Nested U-Net) for 3->3 regression.
    base_ch=32 is a good starting point for 4GB VRAM.
    """
    def __init__(self, in_ch=3, out_ch=3, base_ch=32, depth=4):
        super().__init__()
        ch = [base_ch * (2 ** i) for i in range(depth + 1)]
        self.depth = depth

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder blocks x_{i,0}
        self.enc = nn.ModuleList([ConvBlock(in_ch, ch[0])] +
                                 [ConvBlock(ch[i - 1], ch[i]) for i in range(1, depth + 1)])

        # Decoder/nested blocks x_{i,j}
        # store blocks for each (i,j) with j>=1
        self.blocks = nn.ModuleDict()
        for j in range(1, depth + 1):
            for i in range(0, depth - j + 1):
                in_channels = ch[i] + j * ch[i + 1]
                self.blocks[f"x{i}_{j}"] = ConvBlock(in_channels, ch[i])

        self.final = nn.Conv2d(ch[0], out_ch, kernel_size=1)

    def _up(self, x, target):
        return F.interpolate(x, size=target.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        # x_{i,0}
        x0 = [None] * (self.depth + 1)
        x0[0] = self.enc[0](x)
        for i in range(1, self.depth + 1):
            x0[i] = self.enc[i](self.pool(x0[i - 1]))

        # x_{i,j}
        xs = {(i, 0): x0[i] for i in range(self.depth + 1)}
        for j in range(1, self.depth + 1):
            for i in range(0, self.depth - j + 1):
                ups = [self._up(xs[(i + 1, k)], xs[(i, 0)]) for k in range(0, j)]
                cat = torch.cat([xs[(i, 0)]] + ups, dim=1)
                xs[(i, j)] = self.blocks[f"x{i}_{j}"](cat)

        out = self.final(xs[(0, self.depth)])
        return out