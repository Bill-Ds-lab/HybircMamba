import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class MambaPatchMerging(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 3, 2, 1, groups=dim, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU(inplace=True)
        self.pw = nn.Conv2d(dim, out_dim, 1, bias=False)
        self.norm = nn.LayerNorm(out_dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.bn(self.dw(x)))
        x = self.pw(x)
        return self.norm(x.permute(0, 2, 3, 1).contiguous())

