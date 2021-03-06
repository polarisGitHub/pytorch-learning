# -*- coding: utf-8 -*-
import time
from optparse import OptionParser

import json
import codecs
import torch
import torchvision
from torchvision import transforms
import netfactory

from PIL import Image

parser = OptionParser()
parser.add_option("-p", "--predict_dir", dest="predict_dir", help="test file dir", type="string")
parser.add_option("-m", "--model", dest="model", help="model", type="string")
parser.add_option("-n", "--net", dest="net", help="net", type="string")

(options, args) = parser.parse_args()

dataset = torchvision.datasets.ImageFolder(options.predict_dir,
                                           transform=transforms.Compose([
                                               transforms.Grayscale(),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=(0.5,), std=(0.5,)),
                                           ]))
predicted_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

with codecs.open("model/label.json", "r", encoding='utf-8') as f:
    index = json.load(f)

net = netfactory.getNet(options.net)
net.load_state_dict(torch.load(options.model, map_location='cpu'))
net.eval()

for i, data in enumerate(predicted_loader, 0):
    inputs, labels = data
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    right = (predicted == labels).sum().item()

    print("label", index[str(labels.data.numpy()[0])], "predict", index[str(predicted.data.numpy()[0])], right)
    img = Image.fromarray(inputs.data.numpy()[0, 0, :, :] * 255)
    img.show()
    time.sleep(5)
    img.close()
