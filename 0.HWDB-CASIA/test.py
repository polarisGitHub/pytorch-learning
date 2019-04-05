# -*- coding: utf-8 -*-
from optparse import OptionParser

import torch
import torchvision
from torchvision import transforms
from multiprocessing import cpu_count
from lenet import Net as TestNet

parser = OptionParser()
parser.add_option("-t", "--test_dir", dest="test_dir", help="test file dir", type="string")
parser.add_option("-m", "--model", dest="model", help="model", type="string")
parser.add_option("-n", "--net", dest="net", help="net", type="string")

(options, args) = parser.parse_args()

dataset = torchvision.datasets.ImageFolder(options.test_dir,
                                           transform=transforms.Compose([
                                               transforms.Grayscale(),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,)),
                                           ]))
test_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, num_workers=cpu_count())

gpu = torch.cuda.is_available()
print("use gpu", gpu)

if options.net == 'lenet':
    print('use lenet')
    from lenet import Net as TestNet
elif options.net == 'resnet18':
    print('use resnet18')
    from resnet18 import Net as TestNet

net = TestNet()
net.load_state_dict(torch.load(options.model))
net.eval()

if gpu:
    net = net.cuda()
correct = 0
total = 0
for i, (inputs, labels) in enumerate(test_loader, 0):
    if gpu:
        inputs = inputs.cuda()
        labels = labels.cuda()
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)

    total += 128
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the  test images: %d %%' % (100 * correct / total))
