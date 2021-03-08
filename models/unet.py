from torch import nn
from config import *
from Block import Block


class DownSample(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(DownSample, self).__init__()
        self.maxpooling = nn.MaxPool2d(2, stride=2)

        self.convs = nn.Sequential(Block(inplanes, outplanes, 3,
                                         padding=1, bias=True),
                                   Block(outplanes, outplanes, 3,
                                         padding=1, bias=True),
                                   )

    def forward(self, x):
        x = self.maxpooling(x)
        x = self.convs(x)
        return x


class UpSample(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.convs = nn.Sequential(Block(inplanes, outplanes, 3,
                                         padding=1, bias=True),
                                   Block(outplanes, outplanes, 3,
                                         padding=1, bias=True),
                                   )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat((x1, x2), dim=1)
        x = self.convs(x)
        return x


class UNet(nn.Module):
    def __init__(self, out_channels=2):
        super(UNet, self).__init__()

        self.convs = nn.Sequential(Block(1, 32, 3,
                                         padding=1, bias=True),
                                   Block(32, 32, 3,
                                         padding=1, bias=True),
                                   )
        self.down1 = DownSample(32, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)

        self.up1 = UpSample(256+128, 128)
        self.up2 = UpSample(128+64, 64)
        self.up3 = UpSample(64+32, 32)

        self.cls = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # encoder
        x1 = self.convs(x)  # [64, 512, 512]
        x2 = self.down1(x1)  # [128, 256, 256]
        x3 = self.down2(x2)  # [256, 128, 128]
        x4 = self.down3(x3)  # [512, 64, 64]
        # x = self.down4(x)  # [512, 32, 32]

        # decoder
        x = self.up1(x4, x3)  # [256, 128, 128]
        x = self.up2(x, x2)  # [128, 256, 256]
        x = self.up3(x, x1)  # [64, 512, 512]
        x = self.cls(x)

        return x