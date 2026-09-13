import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from ..layers.downsample import MambaPatchMerging
from ..layers.mamba_block import TriBranchMambaBlockV4


class MambaStage(nn.Module):
    def __init__(self, in_dim, out_dim, drop_path=0.0, num_blocks=1, ssm_d_state=8, ssm_ratio=1.0,
                 layer_scale_init=1e-4, ssm_frac=0.5, conv_frac=0.3):
        super().__init__()
        self.down = MambaPatchMerging(in_dim, out_dim)
        self.blocks = nn.ModuleList([
            TriBranchMambaBlockV4(out_dim, drop_path, ssm_ratio, ssm_d_state,
                                  layer_scale_init, ssm_frac, conv_frac)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x = self.down(x)
        for block in self.blocks:
            x = block(x)
        return x