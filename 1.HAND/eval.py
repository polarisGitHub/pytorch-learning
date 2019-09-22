# -*- coding: utf-8 -*-

import torchvision
from PIL import Image
import matplotlib.pyplot as plt

model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
# set it to evaluation mode, as the model behaves differently
# during training and during evaluation
model.eval()


image = Image.open('/Users/hexie/Documents/05.png')
image_tensor = torchvision.transforms.functional.to_tensor(image)

# pass a list of (potentially different sized) tensors
# to the model, in 0-1 range. The model will take care of
# batching them together and normalizing
prediction = model([image_tensor])


plt.imshow(image)
plt.show()
for p in prediction:
    plt.imshow(Image.fromarray(p['masks'][0, 0].mul(255).byte().cpu().numpy()))
    plt.show()
