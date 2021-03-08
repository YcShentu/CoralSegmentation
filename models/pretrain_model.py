import torch
from torch import nn

from models.resnet import resnet34


class PretrainedModel(nn.Module):
    def __init__(self, output_channels=2):
        super(PretrainedModel, self).__init__()

        self.backbone = resnet34()
        self.gpa = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, output_channels)

    def forward(self, x):
        x, _ = self.backbone(x)  # [256, 32, 32]
        x = self.gpa(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x


