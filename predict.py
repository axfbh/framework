import os.path

import numpy as np

from model.modeling import get_model
import torch

import cv2
from omegaconf import OmegaConf
import albumentations as A
from ops.detection.nms import non_max_suppression
from utils.utils import handle_preds
from albumentations.pytorch import ToTensorV2
from ops.transform.resize_maker import ResizeShortLongest
import ops.cv.io as io
from ops.dataset.utils import batch_images
from ops.dataset.voc_dataset import id2name
from ops.detection.postprocess_utils import YoloPostProcess


def setup(args):
    model = get_model(args)
    if os.path.exists(args.model.weights.checkpoints):
        print('loading checkpoint')
        model.load_state_dict(torch.load(args.model.weights.checkpoints)['model'])
    else:
        raise IndexError(f'cannot find the {args.model.weights.checkpoints}')
    model.to(args.device.test)
    return model


def inference(model, image, anchors, post_process, device):
    h, w = image.shape[-2:]
    preds = model(image.to(device))

    # ------- 补充 ---------
    return post_process(preds, anchors, (h, w))


@torch.no_grad()
def predict(model, args):
    model.eval()

    root = args.test.root

    device = args.test.device

    anchors = torch.tensor(args.train.anchors, device=device).view(3, -1, 2)  # (scale, grid_num, point)

    post_process = YoloPostProcess(device, conf_thres=0.2, iou_thres=0.6)

    with open(args.test_dataset.path, 'r') as fp:
        loader = fp.readlines()

    batch_image = []
    images = []
    for i in range(len(loader)):
        image_path = os.path.join(root, loader[i].strip().split(' ')[0])
        image = io.imread(image_path)

        pad_sample = ResizeShortLongest(args.image_size)(image)

        normal_image = A.Normalize()(image=pad_sample['image'])

        tensor_image = ToTensorV2()(image=normal_image['image'])['image']

        batch_image.append(tensor_image)
        images.append(pad_sample['image'].copy())

        if (i + 1) % 5 == 0:
            batch_image = batch_images(batch_image)

            output = inference(model, batch_image, anchors, post_process, device)

            for i, out in enumerate(output):
                if out is not None:
                    for p in out:
                        cv2.rectangle(images[i],
                                      tuple(p[:2].cpu().int().numpy().copy()),
                                      tuple(p[2:4].cpu().int().numpy().copy()),
                                      (0, 255, 0), 1)
                        cv2.putText(images[i],
                                    id2name.get(p[5].item() + 1),
                                    tuple(
                                        p[:2].cpu().int().numpy().copy() + np.array([-5, -5])),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.65,
                                    (0, 0, 255), 2)
                io.show_window('nam', images[i])
            batch_image = []
            images.clear()


def main():
    args = OmegaConf.load("./config/config.yaml")

    model = setup(args)

    predict(model, args)


if __name__ == '__main__':
    main()
