import os
import uuid
import struct
import numpy as np
from PIL import Image


def read_file(f):
    header_size = 10
    while True:
        header = np.fromfile(f, dtype='uint8', count=header_size)
        if not header.size: break
        sample_size = header[0] + (header[1] << 8) + (header[2] << 16) + (header[3] << 24)
        tag_code = header[5] + (header[4] << 8)
        width = header[6] + (header[7] << 8)
        height = header[8] + (header[9] << 8)
        if header_size + width * height != sample_size:
            break
        data = np.fromfile(f, dtype='uint8', count=width * height).reshape((height, width))
        yield data, tag_code


def decode(gnt_file, root_dir, width=72, height=72):
    with open(gnt_file, 'rb') as f:
        for data, tag_code in read_file(f):
            character = struct.pack('>H', tag_code).decode('gb2312')
            img_dir = "%s/%s" % (root_dir, character)
            if not os.path.exists(img_dir):
                os.makedirs(img_dir)
            image = Image.fromarray(data)
            ration = min(width / image.size[0], height / image.size[1])
            resize_width, resize_height = int(image.size[0] * ration), int(image.size[1] * ration)
            image = image.convert('L').resize((resize_width, resize_height), Image.ANTIALIAS)
            resize_image = Image.new('L', (width, height), 255)
            resize_image.paste(image, (abs(image.size[0] - width) // 2, abs(image.size[1] - height) // 2))
            resize_image.save("%s/%s/%s.%s" % (root_dir, character, uuid.uuid4(), 'png'))


decode("./data/1.0test-gb1.gnt", "./data/test")
decode("./data/1.0train-gb1.gnt", "./data/train")
