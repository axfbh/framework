import os

from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
import ops.cv.io as io

np.random.seed(0)


class FileLoader:
    def __init__(self, path, train_ratio=0.9):
        self.train_ratio = train_ratio
        self.image_paths = []
        self.json_paths = []

        self.labels = []
        self.labels2 = []
        self.labels3 = []
        self.get_file(path)
        self.spilt_train_val()

    def get_file(self, path):
        with open(path, 'r') as fp:
            lines = fp.readlines()

        for line in lines:
            path_split = line.split(";")
            self.image_paths.append(path_split[3].strip())
            self.json_paths.append(path_split[4].strip())
            self.labels.append(int(path_split[0]))
            self.labels2.append(int(path_split[1]))
            self.labels3.append(int(path_split[2]))

        self.image_paths = np.array(self.image_paths)
        self.json_paths = np.array(self.json_paths)
        # -------------- 有没有字类别 ----------------
        self.labels = np.array(self.labels)
        # -------------- 图片类别 ---------------
        self.labels2 = np.array(self.labels2)
        # -------------- json类别 -------------
        self.labels3 = np.array(self.labels3)

    def spilt_train_val(self):
        train_len = len(self.image_paths)
        all_id = range(train_len)
        self.train_id = np.random.choice(all_id, int(train_len * self.train_ratio))
        self.val_id = np.setdiff1d(all_id, self.train_id)


class MyDataSetTrain(Dataset):
    def __init__(self, file_loader, image_size, transform):
        super().__init__()
        self.file_loader = file_loader
        self.transform = transform
        self.image_size = image_size

        self.image_paths = file_loader.image_paths[file_loader.train_id]
        self.json_paths = file_loader.json_paths[file_loader.train_id]
        # -------------- 有没有字类别 ----------------
        self.labels = file_loader.labels[file_loader.train_id]
        # -------------- 图片类别 ---------------
        self.labels2 = file_loader.labels2[file_loader.train_id]
        # -------------- json类别 -------------
        self.labels3 = file_loader.labels3[file_loader.train_id]

    def __getitem__(self, idx):
        return 0

    def __len__(self):
        return len(self.image_paths)


class MyDataSetVal(Dataset):
    def __init__(self, file_loader, image_size):
        super().__init__()
        self.file_loader = file_loader
        self.image_size = image_size

        self.image_paths = file_loader.image_paths[file_loader.val_id]
        self.json_paths = file_loader.json_paths[file_loader.val_id]
        # -------------- 有没有字类别 ----------------
        self.labels = file_loader.labels[file_loader.val_id]
        # -------------- 图片类别 ---------------
        self.labels2 = file_loader.labels2[file_loader.val_id]
        # -------------- json类别 -------------
        self.labels3 = file_loader.labels3[file_loader.val_id]

    def make_tensor(self, image, image_size):
        resize_image = cv2.resize(image, (image_size, image_size))
        tensor_image = A.Normalize()(image=resize_image.copy())['image']
        tensor_image = ToTensorV2()(image=tensor_image)['image'].unsqueeze(0)
        return tensor_image, resize_image

    def __getitem__(self, idx):
        return 0

    def __len__(self):
        return len(self.image_paths)


def get_loader(args):
    file_loader = FileLoader(path=args.train_dataset.path)

    train_transform = A.Compose([
        A.Resize(args.image_size, args.image_size),
        A.HorizontalFlip(p=0.2),
        A.VerticalFlip(p=0.3),
        A.GaussNoise(p=0.3),  # 将高斯噪声应用于输入图像。
        # A.OneOf([
        #     A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
        #     A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
        #     A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
        # ], p=0.3),
        # 随机应用仿射变换：平移，缩放和旋转输入
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0., rotate_limit=90, p=0.4, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(brightness_limit=-0.2, p=0.4),
        A.Normalize(),
        ToTensorV2()
    ])

    train_dataset = MyDataSetTrain(file_loader,
                                   image_size=args.image_size,
                                   transform=train_transform)

    val_dataset = MyDataSetVal(file_loader,
                               image_size=args.image_size)

    nw = min(3, args.train.batch_size, os.cpu_count())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train.batch_size,
                              shuffle=False,
                              num_workers=nw,
                              drop_last=True)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=args.val_dataset.batch_size,
                            shuffle=False,
                            num_workers=nw,
                            drop_last=True)

    return train_loader, val_loader
