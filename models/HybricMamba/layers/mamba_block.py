import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from .ssm import SS2DBiScan

try:
    SSMODE = "sscore"
    import selective_scan_cuda_core
except Exception:
    SSMODE = "mamba_ssm"
    try:
        import selective_scan_cuda
    except Exception:
        selective_scan_cuda = None

class TriBranchMambaBlockV4(nn.Module):
    def __init__(self, hidden_dim, drop_path=0.0, ssm_ratio=1.0, ssm_d_state=8,
                 layer_scale_init=1e-4, ssm_frac=0.5, conv_frac=0.3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        self.ssm_dim = max(4, int(round(hidden_dim * ssm_frac / 2)) * 2)
        self.conv_dim = max(4, int(round(hidden_dim * conv_frac / 2)) * 2)
        self.id_dim = hidden_dim - self.ssm_dim - self.conv_dim

        perm = torch.randperm(hidden_dim)
        inv_perm = torch.argsort(perm)
        self.register_buffer("perm", perm, persistent=True)
        self.register_buffer("inv_perm", inv_perm, persistent=True)

        self.ssm = SS2DBiScan(d_model=self.ssm_dim, d_state=ssm_d_state, ssm_ratio=ssm_ratio)

        half = self.conv_dim // 2
        self.mk_conv3 = nn.Conv2d(half, half, kernel_size=3, padding=1, groups=half, bias=False)
        self.mk_conv5 = nn.Conv2d(self.conv_dim - half, self.conv_dim - half, kernel_size=5, padding=2,
                                  groups=self.conv_dim - half, bias=False)
        self.mk_bn = nn.BatchNorm2d(self.conv_dim)
        self.mk_act = nn.SiLU(inplace=True)

        self.drop_path = DropPath(drop_path)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(hidden_dim))

    def forward(self, x):
        shortcut = x
        x_norm = self.norm(x)[..., self.perm]

        x_ssm, x_conv, x_id = torch.split(x_norm, [self.ssm_dim, self.conv_dim, self.id_dim], dim=-1)

        out_ssm = self.ssm(x_ssm)

        x_conv_cf = x_conv.permute(0, 3, 1, 2).contiguous()
        half = self.conv_dim // 2
        c3, c5 = torch.split(x_conv_cf, [half, self.conv_dim - half], dim=1)
        out_conv_cf = self.mk_act(self.mk_bn(torch.cat([self.mk_conv3(c3), self.mk_conv5(c5)], dim=1)))
        out_conv = out_conv_cf.permute(0, 2, 3, 1).contiguous()

        out = torch.cat([out_ssm, out_conv, x_id], dim=-1)[..., self.inv_perm]
        return shortcut + self.drop_path(self.gamma * out)

