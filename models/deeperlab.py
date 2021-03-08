import torch
from torch import nn
from torch.nn import functional as F

from models.resnet import resnet34
from Block import Block


class Space2Depth(nn.Module):
    def __init__(self, stride=1):
        super(Space2Depth, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.shape
        assert (H % self.stride == 0 and W % self.stride == 0), 'wrong tensor dimensions'
        h = H // self.stride
        w = W // self.stride
        x = x.view(B, C, h, self.stride, w, self.stride).transpose(3, 4).contiguous()
        x = x.view(B, C, h*w, self.stride*self.stride).transpose(2, 3).contiguous()
        x = x.view(B, C, self.stride*self.stride, h*w).transpose(1, 2).contiguous()
        x = x.view(B, self.stride*self.stride*C, h, w)
        return x


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

        self.aspp_5 = Block(256*5, 256, kernel=1)

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


class DeeperLabC(nn.Module):
    def __init__(self, output_channels=2):
        super(DeeperLabC, self).__init__()

        self.backbone = resnet34()

        # for high level features
        self.aspp = ASPP()

        # for low level features
        self.low_level_f_convs = Block(64, 32, kernel=1)
        self.s2d = Space2Depth(stride=4)

        # after concat
        self.convs_channel = 128
        self.convs1 = Block(256+512, self.convs_channel,
                            kernel=7, stride=1,
                            padding=3)
        self.convs2 = Block(self.convs_channel, self.convs_channel,
                            kernel=7, stride=1,
                            padding=3)
        self.d2s = nn.PixelShuffle(upscale_factor=4)  # [128/16, 128, 128]

        # segmentation
        self.convs3 = Block(self.convs_channel//16, self.convs_channel//16,
                            kernel=7, stride=1,
                            padding=3)
        self.convs4 = nn.Conv2d(self.convs_channel//16, output_channels,
                                kernel_size=1)

    def forward(self, x):
        hlf, llf = self.backbone(x)  # [256, 32, 32], [64, 128, 128]

        hlf = self.aspp(hlf)

        llf = self.low_level_f_convs(llf)
        llf = self.s2d(llf)

        x = torch.cat((llf, hlf), dim=1)

        x = self.convs1(x)
        x = self.convs2(x)
        x = self.d2s(x)

        # start segmentation
        x = self.convs3(x)
        x = self.convs4(x)

        x = F.interpolate(x, scale_factor=4, mode='bilinear')  # heatmaps

        return x


