import numpy as np
import torch
import math


def batch_images(images, size_divisible=32):
    max_h, max_w = np.array([[img.shape[1], img.shape[2]] for img in images], dtype=int).max(0)

    stride = float(size_divisible)
    max_h = int(math.ceil(float(max_h) / stride) * stride)
    max_w = int(math.ceil(float(max_w) / stride) * stride)

    batched_imgs = torch.zeros((len(images), 3, max_h, max_w), dtype=images[0].dtype)

    for i in range(len(images)):
        img = images[i]
        batched_imgs[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    return batched_imgs


def batch_labels(labels):
    max_h, max_w = np.array([[l.shape[0], l.shape[1]] for l in labels], dtype=int).max(0)

    batched_labels = torch.full((len(labels), max_h, max_w), dtype=labels[0].dtype, fill_value=-999999)

    for i, l in enumerate(labels):
        if l.shape[0] > 0:
            batched_labels[i, : l.shape[0], : l.shape[1]].copy_(l)
            batched_labels[i, :, 0] = i
    return batched_labels


def detect_collate_fn(batch):
    images, labels = zip(*batch)

    batched_imgs = batch_images(images)

    batched_labels = batch_labels(labels)

    return batched_imgs, batched_labels