from torch import nn
from torch.nn import functional as F

from models.resnet import resnet34
from Block import Block


class FCN4x(nn.Module):
    def __init__(self, out_channels=2):
        super(FCN4x, self).__init__()
        # backbone -- resnet
        # which is slightly different from the original FCN
        backbone = resnet34()
        self.block1 = nn.Sequential(backbone.conv1,
                                    backbone.bn1,
                                    backbone.relu,
                                    backbone.maxpool)
        self.block2 = backbone.layer1
        self.block3 = backbone.layer2
        self.block4 = backbone.layer3
        self.convs = nn.Sequential(Block(256, 512, kernel=3, padding=1),
                                   Block(512, 512, kernel=3, padding=1))

        # prediction at each scale
        self.pred_x_16 = nn.Conv2d(512, out_channels, kernel_size=1)
        self.pred_x_8 = nn.Conv2d(128, out_channels, kernel_size=1)
        self.pred_x_4 = nn.Conv2d(64, out_channels, kernel_size=1)

        # upsampling
        self.convs5 = Block(out_channels, out_channels,
                            kernel=3, stride=1,
                            padding=1, bias=False)
        self.convs4 = Block(out_channels, out_channels,
                            kernel=3, stride=1,
                            padding=1, bias=False)
        self.convs3 = Block(out_channels, out_channels,
                            kernel=3, stride=1,
                            padding=1, bias=False)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x_4 = x
        x = self.block3(x)
        x_8 = x
        x = self.block4(x)
        x = self.convs(x)

        # upsampling
        x = self.pred_x_16(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.convs5(x)

        x += self.pred_x_8(x_8)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.convs4(x)

        x += self.pred_x_4(x_4)
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.convs3(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        return x


if __name__ == '__main__':
    fcn = FCN4x()

    print('Loading fcn')






