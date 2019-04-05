# -*- coding: utf-8 -*-

from lenet import Net as lenet
from resnet18 import Net as resnet18


def getNet(net):
    if net == 'lenet':
        return lenet()
    elif net == 'resnet18':
        return resnet18()
