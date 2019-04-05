# -*- coding: utf-8 -*-
from torch import nn
import torchvision.models as models
from torchsummary import summary
from torchvision.models.resnet import Bottleneck


class Net(models.ResNet):
    def __init__(self):
        super(Net, self).__init__(Bottleneck, [3, 4, 6, 3], num_classes=3740)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)


if __name__ == "__main__":
    summary(Net(), (1, 72, 72))
