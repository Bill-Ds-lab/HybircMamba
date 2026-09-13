import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from ..ops.scan_transform import bi_selective_scan

try:
    SSMODE = "sscore"
    import selective_scan_cuda_core
except Exception:
    SSMODE = "mamba_ssm"
    try:
        import selective_scan_cuda
    except Exception:
        selective_scan_cuda = None


class SS2DBiScan(nn.Module):
    def __init__(self, d_model=64, d_state=8, ssm_ratio=1.0, act_layer=nn.SiLU, d_conv=3, conv_bias=True, dropout=0.0):
        super().__init__()
        factory_kwargs = {"device": None, "dtype": None}
        d_expand = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_model = d_model
        self.d_expand = d_expand

        self.out_norm = nn.LayerNorm(d_expand)
        self.K = 2  # Bi-directional scan

        self.in_proj = nn.Linear(d_model, d_expand * 2, bias=False, **factory_kwargs)
        self.act = act_layer()

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(in_channels=d_expand, out_channels=d_expand, groups=d_expand, bias=conv_bias,
                                    kernel_size=d_conv, padding=(d_conv - 1) // 2, **factory_kwargs)

        self.x_proj = [nn.Linear(d_expand, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs) for _ in range(self.K)]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, d_expand, **factory_kwargs) for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, d_expand)
        self.Ds = self.D_init(d_expand)
        self.out_proj = nn.Linear(d_expand, d_model, bias=False, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5
        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(torch.rand(d_inner, **factory_kwargs) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad(): dt_proj.bias.copy_(inv_dt)
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner):
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(d_inner, 1)
        A_log = nn.Parameter(torch.log(A))
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner):
        D = nn.Parameter(torch.ones(d_inner))
        D._no_weight_decay = True
        return D

    def forward(self, x: torch.Tensor):
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)
        x_branch = x_branch.permute(0, 3, 1, 2).contiguous()
        x_branch = self.act(self.conv2d(x_branch))

        A_logs = self.A_logs.repeat(self.K, 1)
        Ds = self.Ds.repeat(self.K)

        y = bi_selective_scan(x_branch, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias, A_logs,
                              Ds, self.out_norm, nrows=1, delta_softplus=True)
        y = y * self.act(z)
        return self.dropout(self.out_proj(y))