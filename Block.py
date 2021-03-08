from torch import nn


class Block(nn.Module):
    def __init__(self, in_plane, out_plane, kernel,
                 stride=1, padding=0, dilation=1,
                 groups=1, bias=False, drop_out=False):
        super(Block, self).__init__()

        self.conv2d = nn.Conv2d(in_plane, out_plane,
                                kernel_size=kernel, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_plane)
        self.relu = nn.ReLU()

        self.drop_out = drop_out
        if drop_out:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu(x)

        if self.drop_out:
            x = self.dropout(x)

        return x
