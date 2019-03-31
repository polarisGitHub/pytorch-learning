# -*- coding: utf-8 -*-

from optparse import OptionParser

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from multiprocessing import cpu_count
from lenet import Net as TrainNet

parser = OptionParser()
parser.add_option("-t", "--train_dir", dest="train_dir", help="train file dir", type="string")
(options, args) = parser.parse_args()

dataset = torchvision.datasets.ImageFolder(options.train_dir,
                                           transform=transforms.Compose([
                                               transforms.Grayscale(),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,)),
                                           ]))
train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=cpu_count())

net = TrainNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2 == 1:  # print every 2 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2))
            running_loss = 0.0

print('Finished Training')
