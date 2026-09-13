
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

from .ConvNet_ultils import SEAttention,MaxPoolingChannel,AvgPoolingChannel
from .VSSBlock_ultils import SS2D, Mlp


class Super_Mamba(nn.Module):
    def __init__(self, dims=3, depth=6, num_classes=43):
        super().__init__()
        self.depth = depth
        self.preembd = ConvNet()
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.depth+1)]
        self.num_features = dims[-1]
        self.dims = dims
        self.layers = nn.ModuleList()
        for i_layer in range(self.depth):
            downsample = PatchMerging2D(
                self.dims[i_layer],
                self.dims[i_layer + 1],
                norm_layer=nn.LayerNorm,
            )
            vss_block = VSSBlock(hidden_dim=self.dims[i_layer+1])
            self.layers.append(downsample)
            self.layers.append(vss_block)

        self.classifier = nn.Sequential(OrderedDict(
            norm=nn.LayerNorm(self.num_features),  # B,H,W,C
            permute=Permute(0, 3, 1, 2),
            avgpool=nn.AdaptiveAvgPool2d(1),
            flatten=nn.Flatten(1),
            head=nn.Linear(self.num_features, num_classes),
        ))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.preembd(x)
        x = x.permute(0, 2, 3, 1)
        for layers in self.layers:
            x = layers(x)
        # print(x.shape)
        x = self.classifier(x)
        return x



class PatchMerging2D(nn.Module):
    def __init__(self, dim, out_dim=-1, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, (2 * dim) if out_dim < 0 else out_dim, bias=False)
        self.norm = norm_layer(4 * dim)

    @staticmethod
    def _patch_merging_pad(x: torch.Tensor):
        H, W, _ = x.shape[-3:]
        if (W % 2 != 0) or (H % 2 != 0):
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        x0 = x[..., 0::2, 0::2, :]  # ... H/2 W/2 C
        x1 = x[..., 1::2, 0::2, :]  # ... H/2 W/2 C
        x2 = x[..., 0::2, 1::2, :]  # ... H/2 W/2 C
        x3 = x[..., 1::2, 1::2, :]  # ... H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # ... H/2 W/2 4*C
        return x

    def forward(self, x):
        x = self._patch_merging_pad(x)
        x = self.norm(x)
        x = self.reduction(x)
        return x

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 96,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_simple_init=False,
            forward_type="v2",
            # =============================
            mlp_ratio=0.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm = norm_layer(hidden_dim)
        self.op = SS2D(
            d_model=hidden_dim,
            d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            ssm_rank_ratio=ssm_rank_ratio,
            dt_rank=ssm_dt_rank,
            act_layer=ssm_act_layer,
            # ==========================
            d_conv=ssm_conv,
            conv_bias=ssm_conv_bias,
            # ==========================
            dropout=ssm_drop_rate,
            # bias=False,
            # ==========================
            # dt_min=0.001,
            # dt_max=0.1,
            # dt_init="random",
            # dt_scale="random",
            # dt_init_floor=1e-4,
            simple_init=ssm_simple_init,
            # ==========================
            forward_type=forward_type,
        )
        self.drop_path = DropPath(drop_path)

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            print("mlp_hidden_dim", mlp_hidden_dim)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                           drop=mlp_drop_rate, channels_first=False)

    def _forward(self, input: torch.Tensor):
        x = input + self.drop_path(self.op(self.norm(input)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x

    def forward(self, input: torch.Tensor):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input)
        else:
            return self._forward(input)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # 第一层卷积
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU()
        )

        # 第二层卷积
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU()

        )

        # 第三层卷积
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)

        )
        self.se = SEAttention()

        self.maxpool = MaxPoolingChannel()
        self.avgpool = AvgPoolingChannel()

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1),
            nn.ReLU(),
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, padding=2)
        )

    def forward(self, x):
        chaneel_1_max_pool = self.maxpool(x)
        desired_size = (x.size(2), x.size(3))
        channel_1_max_pool_out = F.interpolate(chaneel_1_max_pool, size=desired_size, mode='bilinear',
                                               align_corners=False)

        channel_2_1 = self.conv4(x)
        # print("channel_2_1", channel_2_1.shape)
        channel_2_2 = self.conv5(x)
        # print("channel_2_2", channel_2_2.shape)
        channel_2_3 = self.conv6(x)
        # print("channel_2_3", channel_2_3.shape)

        channel_2_sum = 0.2 * channel_2_1 + 0.6 * channel_2_2 + 0.2 * channel_2_3


        channel_2_x_1 = self.conv1(x)
        channel_2_x_2 = self.conv2(channel_2_x_1)
        channel_2_x_3 = self.conv3(channel_2_x_2)
        channel_2_x_4 = self.se(channel_2_x_3)

        channel_2_total = channel_2_x_4 * 0.6 + channel_2_x_3 * 0.4
        channel_2_total_avg_pool = self.avgpool(channel_2_total)
        desired_size = (x.size(2), x.size(3))
        channel_2_avg_pool_out = F.interpolate(channel_2_total_avg_pool, size=desired_size, mode='bilinear',
                                               align_corners=False)

        channel_3 = x

        total_3_channel = 0 * channel_1_max_pool_out + 1 * channel_2_avg_pool_out + 0 * channel_3
        return total_3_channel



