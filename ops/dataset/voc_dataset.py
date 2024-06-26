import torch
import xml.etree.ElementTree as ET
import os

import numpy as np

from torch.utils.data import Dataset
import ops.cv.io as io

CLASSES_NAME = (
    "__background__",
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

name2id = dict(zip(CLASSES_NAME, range(len(CLASSES_NAME))))

id2name = {v: k for k, v in name2id.items()}


class VOCDetection(Dataset):

    def __init__(self, root_dir, image_set, args, transform=None):
        super(VOCDetection, self).__init__()

        self._annopath = os.path.join(root_dir, "Annotations", "%s.xml")
        self._imgpath = os.path.join(root_dir, "JPEGImages", "%s.jpg")
        self._imgsetpath = os.path.join(root_dir, "ImageSets", "Main", "%s.txt")

        self.image_size = args.image_size

        self.transform = transform

        with open(self._imgsetpath % image_set) as f:
            self.img_ids = f.readlines()

        if image_set in ['train', 'trainval', 'val']:
            self.img_ids = [x.strip() for x in self.img_ids]
        else:
            img_ids_flag = [x.strip().split(' ') for x in self.img_ids]
            self.img_ids = [x[0] for x in img_ids_flag if x[1] != '-1']

    def __len__(self):
        return len(self.img_ids)

    def __annotations(self, img_id):
        anno = ET.parse(self._annopath % img_id).getroot()
        bbox_params = []
        classes = []
        for obj in anno.iter("object"):
            _box = obj.find("bndbox")
            box = [
                _box.find("xmin").text,
                _box.find("ymin").text,
                _box.find("xmax").text,
                _box.find("ymax").text,
            ]
            bbox_params.append(box)

            name = obj.find("name").text.lower().strip()
            classes.append(name2id[name])

        bbox_params = np.array(bbox_params, dtype=np.float32)
        return bbox_params, classes

    def __getitem__(self, idx):
        image = io.imread(self._imgpath % self.img_ids[idx])

        bbox_params, classes = self.__annotations(self.img_ids[idx])

        return image, bbox_params, classes
