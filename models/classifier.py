import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, out_channels=2):
        super(Classifier, self).__init__()
        # classification head
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels, out_channels, False)  # data shape (2, 2) --> (out, in)

    def forward(self, x):
        x = self.global_avg_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x
