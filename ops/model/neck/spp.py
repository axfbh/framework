import torch.nn as nn
import torch
from torchvision.ops.misc import Conv2dNormActivation
from functools import partial


class SPP(nn.Module):
    def __init__(self, ksizes=(5, 9, 13)):
        """
            SpatialPyramidPooling 空间金字塔池化
        """
        super(SPP, self).__init__()
        self.make_layers = nn.ModuleList([nn.MaxPool2d(kernel_size=k, stride=1, padding=(k - 1) // 2) for k in ksizes])

    def forward(self, x):
        return torch.cat([x] + [m(x) for m in self.make_layers], 1)


class SPPCSPC(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=0.5, ksizes=(5, 9, 13), activation_layer=nn.ReLU):
        super(SPPCSPC, self).__init__()

        CBS = partial(Conv2dNormActivation,
                      bias=False,
                      inplace=False,
                      norm_layer=nn.BatchNorm2d,
                      activation_layer=activation_layer)

        mid_ch = int(2 * out_ch * expand_ratio)

        self.cv1 = nn.Sequential(
            CBS(in_ch, mid_ch, 1),
            CBS(mid_ch, mid_ch, 3),
            CBS(mid_ch, mid_ch, 1),
        )

        self.spp = SPP(ksizes)

        self.cv2 = nn.Sequential(
            CBS(mid_ch * 4, mid_ch, 1),
            CBS(mid_ch, mid_ch, 3),
        )

        self.cv3 = CBS(in_ch, mid_ch, 1)

        self.cv4 = CBS(mid_ch * 2, out_ch, 1)

    def forward(self, x):
        x1 = self.cv1(x)
        x1 = self.spp(x1)
        x1 = self.cv2(x1)

        x2 = self.cv3(x)

        x = torch.cat([x1, x2], dim=1)

        return self.cv4(x)
