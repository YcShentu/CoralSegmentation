import torch
from torch import nn
from torch.nn import functional as F

from models.resnet import resnet34
from Block import Block


class ASPP(nn.Module):
    """
    Atrous spatial pyramid pooling, ASPP
    """

    def __init__(self):
        super(ASPP, self).__init__()

        self.aspp_1 = Block(256, 256,
                            kernel=1, stride=1,
                            padding=0, dilation=1)

        self.aspp_2 = Block(256, 256,
                            kernel=3, stride=1,
                            padding=6, dilation=6)

        self.aspp_3 = Block(256, 256,
                            kernel=3, stride=1,
                            padding=12, dilation=12)

        self.aspp_4 = Block(256, 256,
                            kernel=3, stride=1,
                            padding=18, dilation=18)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.aspp_5 = Block(256 * 5, 256, kernel=1)

    def forward(self, x):
        x1 = self.aspp_1(x)
        x2 = self.aspp_2(x)
        x3 = self.aspp_3(x)
        x4 = self.aspp_4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, (32, 32), mode='bilinear')

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.aspp_5(x)

        return x


class DeepLabV3Plus(nn.Module):
    def __init__(self, out_channels=2):
        super(DeepLabV3Plus, self).__init__()

        self.backbone = resnet34()

        # high level features
        self.aspp = ASPP()
        self.dropout1 = nn.Dropout(0.5)

        # low level features
        self.conv1 = Block(64, 32, kernel=1)

        # after concat
        self.convs2 = nn.Sequential(Block(256 + 32, 64, kernel=3,
                                          padding=1, drop_out=True),
                                    Block(64, 32, kernel=3,
                                          padding=1, drop_out=True),

                                    )

        self.conv3 = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        hlf, llf = self.backbone(x)  # [256, 32, 32], [64, 128, 128]

        hlf = self.aspp(hlf)
        hlf = self.dropout1(hlf)
        hlf = F.interpolate(hlf, scale_factor=4, mode='bilinear', align_corners=True)

        llf = self.conv1(llf)

        x = torch.cat((hlf, llf), dim=1)

        x = self.convs2(x)
        x = self.conv3(x)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)

        return x