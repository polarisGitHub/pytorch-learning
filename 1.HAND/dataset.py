# -*- coding: utf-8 -*-
import torch

import cv2
import numpy as np
import os
from scipy.io import loadmat


def polygon_to_box(polygon):
    x = list(map(lambda l: l[0], polygon))
    y = list(map(lambda l: l[1], polygon))
    return [np.min(x), np.min(y), np.max(x), np.max(y)]


class HandDataset(object):
    def __init__(self, base_path, transforms):
        self.base_path = base_path
        self.transforms = transforms
        self.dataset = []

        for folder in os.listdir(self.base_path):
            if folder.startswith("."):
                continue
            folder_path = self.base_path + folder
            files = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('jpg')]
            files.sort()

            video = loadmat(os.path.join(folder_path, 'polygons.mat'))
            polygons = video['polygons'][0]
            for i, file in enumerate(files):
                polygon_list = []
                for polygon in polygons[i]:
                    p = polygon.astype(np.int32)
                    if p.size > 0:
                        polygon_list.append(p)
                if len(polygon_list) > 0:
                    self.dataset.append({
                        "file": folder_path + "/" + file + ".jpg",
                        "polygons": polygon_list
                    })

    def __getitem__(self, idx):
        file, polygons = self.dataset[idx]['file'], self.dataset[idx]['polygons']
        image = cv2.imread(file)
        num_objs, masks, boxes = len(polygons), np.zeros(image.shape[:2], dtype="uint8"), []

        cv2.polylines(masks, polygons, 1, 255)
        cv2.fillPoly(masks, polygons, 255)
        for index, polygon in enumerate(polygons):
            boxes.append(polygon_to_box(polygon))
        # for box in boxes:
        #     image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
        # cv2.imshow("", image)
        # cv2.waitKey(0)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        target = {
            "boxes": boxes,
            "labels": torch.ones((num_objs,), dtype=torch.int64),
            "masks": torch.as_tensor(masks, dtype=torch.uint8),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros((num_objs,), dtype=torch.int64)

        }
        if self.transforms is not None:
            image = self.transforms(image)
        return image, target

    def __len__(self):
        return len(self.dataset)

# dataset = HandDataset("/Volumes/data/hand_dectection/egohands_data/_LABELLED_SAMPLES/", None)
# dataset[300]
