import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from .attention import SimAM
class MBConvSimAM(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=4, kernel_size=3, drop_path=0.0):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_res = in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False), nn.BatchNorm2d(mid), nn.SiLU(inplace=True)]
        else:
            mid = in_ch
        layers += [nn.Conv2d(mid, mid, kernel_size, padding=kernel_size // 2, groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.SiLU(inplace=True)]
        self.body = nn.Sequential(*layers)
        self.simam = SimAM()
        self.proj = nn.Sequential(nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        self.drop_path = DropPath(drop_path)

    def forward(self, x):
        y = self.proj(self.simam(self.body(x)))
        return x + self.drop_path(y) if self.use_res else y
