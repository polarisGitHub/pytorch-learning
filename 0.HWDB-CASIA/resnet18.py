# -*- coding: utf-8 -*-
from torch import nn
import torchvision.models as models
from torchsummary import summary


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.net = models.resnet18()
        self.net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.net.fc = nn.Linear(512, 3740)

    def forward(self, x):
        return self.net.forward(x)


if __name__ == "__main__":
    summary(Net(), (1, 72, 72))
