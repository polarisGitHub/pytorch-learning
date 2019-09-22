# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
from scipy.io import loadmat


class HandDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        img, target = None, None
        return img, target

    def __len__(self):
        return len(self.imgs)


folder_path = "/Volumes/data/hand_dectection/egohands_data/_LABELLED_SAMPLES/JENGA_COURTYARD_H_B"

files = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('jpg')]
files.sort()

video = loadmat(os.path.join(folder_path, 'polygons.mat'))
polygons = video['polygons'][0]  # there are 100*4 entries in polygons
for i, file in enumerate(files):
    path = folder_path + "/" + file + ".jpg"
    image = cv2.imread(path)
    width, height = np.size(image, 1), np.size(image, 2)
    for point_list in polygons[i]:
        if point_list.size > 0:
            point_list_int = [point_list.astype(np.int32)]
            mask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.polylines(mask, point_list_int, 1, 255)
            cv2.fillPoly(mask, point_list_int, 255)
            masked = cv2.bitwise_and(image, image, mask=mask)
            cv2.imshow("Mask to Image", masked)
            cv2.waitKey(0)
