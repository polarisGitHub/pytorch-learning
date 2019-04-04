# -*- coding: utf-8 -*-

from optparse import OptionParser

import codecs
import json
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from multiprocessing import cpu_count
from lenet import Net as TrainNet

gpu = torch.cuda.is_available()
print("use gpu", gpu)

parser = OptionParser()
parser.add_option("-t", "--train_dir", dest="train_dir", help="train file dir", type="string")
parser.add_option("-e", "--epoches", dest="epoches", help="epoches", type="int")
parser.add_option("-b", "--batch_size", dest="batch_size", help="batch_size", type="int")
(options, args) = parser.parse_args()

transform = transforms.Compose(
    [transforms.Grayscale(), transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
dataset = torchvision.datasets.ImageFolder(options.train_dir, transform=transform)

with codecs.open('model/label.json', "w", encoding='utf-8') as f:
    label = dataset.class_to_idx
    json.dump(dict(zip(label.values(), label.keys())), f, ensure_ascii=False)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True,
                                           num_workers=cpu_count())

net = TrainNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

if gpu:
    net = net.cuda()
    criterion = criterion.cuda()

for epoch in range(options.epoches):
    running_loss, correct = 0.0, 0

    for i, (inputs, labels) in enumerate(train_loader, 0):
        if gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f correct: %.3f' % (
                epoch, i, running_loss / 100, correct / (100 * options.batch_size)
            ))
            running_loss = 0.0
            correct = 0

torch.save(net.state_dict(), 'model/lenet')
print('Finished Training')
