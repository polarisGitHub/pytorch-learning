# -*- coding: utf-8 -*-
import os
from PIL import Image
from scipy.io import loadmat


class HandDataset(object):
    def __init__(self, base_path, transforms):
        self.base_path = base_path
        self.transforms = transforms
        self.dataset = []

        for folder in os.listdir(self.base_path):
            folder_path = self.base_path + folder
            files = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('jpg')]
            files.sort()

            video = loadmat(os.path.join(folder_path, 'polygons.mat'))
            polygons = video['polygons'][0]
            for i, file in enumerate(files):
                self.dataset.append({
                    "file": folder_path + "/" + file + ".jpg",
                    "polygons": polygons[i]
                })

    def __getitem__(self, idx):
        img, target = None, None
        file, polygons = self.dataset[idx]['file'], self.dataset[idx]['polygons']
        image = Image.open(file).convert("RGB")

        # target = {}
        # target["boxes"] = boxes
        # target["labels"] = labels
        # target["masks"] = masks
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        #
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.dataset)

# files = [os.path.splitext(f)[0] for f in os.listdir(folder_path) if f.endswith('jpg')]
# files.sort()
#
# video = loadmat(os.path.join(folder_path, 'polygons.mat'))
# polygons = video['polygons'][0]  # there are 100*4 entries in polygons
# for i, file in enumerate(files):
#     path = folder_path + "/" + file + ".jpg"
#     image = cv2.imread(path)
#     width, height = np.size(image, 1), np.size(image, 2)
#     for point_list in polygons[i]:
#         if point_list.size > 0:
#             point_list_int = [point_list.astype(np.int32)]
#             mask = np.zeros(image.shape[:2], dtype="uint8")
#             cv2.polylines(mask, point_list_int, 1, 255)
#             cv2.fillPoly(mask, point_list_int, 255)
#             masked = cv2.bitwise_and(image, image, mask=mask)
#             cv2.imshow("Mask to Image", masked)
#             cv2.waitKey(0)
