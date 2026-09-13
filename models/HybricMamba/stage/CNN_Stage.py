import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from ..layers.conv_blocks import MBConvSimAM


class CNNStage(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=4, drop_path=0.0, num_blocks=1):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, 2, 1, groups=in_dim, bias=False),
            nn.BatchNorm2d(in_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim)
        )
        self.blocks = nn.Sequential(*[
            MBConvSimAM(out_dim, out_dim, expand_ratio=expand_ratio, drop_path=drop_path)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        return self.blocks(self.down(x))
