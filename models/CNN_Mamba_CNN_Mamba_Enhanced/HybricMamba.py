import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

try:
    SSMODE = "sscore"
    import selective_scan_cuda_core
except Exception:
    SSMODE = "mamba_ssm"
    try:
        import selective_scan_cuda
    except Exception:
        selective_scan_cuda = None


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        if u.stride(-1) != 1: u = u.contiguous()
        if delta.stride(-1) != 1: delta = delta.contiguous()
        if D is not None: D = D.contiguous()
        if B.stride(-1) != 1: B = B.contiguous()
        if C.stride(-1) != 1: C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        if SSMODE == "mamba_ssm":
            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        else:
            out, x, *rest = selective_scan_cuda_core.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1: dout = dout.contiguous()

        if SSMODE == "mamba_ssm":
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus, False
            )
        else:
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda_core.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
            )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
        return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


class BiScan2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        B, C, H, W = x.shape
        ctx.shape = (B, C, H, W)
        xs = x.new_empty((B, 2, C, H * W))
        xs[:, 0] = x.flatten(2, 3)
        xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
        return xs

    @staticmethod
    def backward(ctx, ys: torch.Tensor):
        B, C, H, W = ctx.shape
        L = H * W
        y0 = ys[:, 0].view(B, C, H, W)
        y1 = ys[:, 1].view(B, C, W, H).transpose(dim0=2, dim1=3)
        return (y0 + y1).contiguous().view(B, C, H, W)


class BiMerge2D(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor):
        B, K, D, H, W = ys.shape
        ctx.shape = (H, W)
        y0 = ys[:, 0]
        y1 = ys[:, 1].transpose(dim0=2, dim1=3).contiguous()
        return y0 + y1

    @staticmethod
    def backward(ctx, x: torch.Tensor):
        H, W = ctx.shape
        B, C, H_x, W_x = x.shape
        xs = x.new_empty((B, 2, C, H, W))
        xs[:, 0] = x
        xs[:, 1] = x.transpose(dim0=2, dim1=3).contiguous()
        return xs  # Trả về 1 tensor ứng với 1 đầu vào ys ở forward


def bi_selective_scan(x, x_proj_weight, x_proj_bias, dt_projs_weight, dt_projs_bias, A_logs, Ds, out_norm, nrows=-1,
                      delta_softplus=True, to_dtype=True):
    B, D, H, W = x.shape
    D, N = A_logs.shape
    K, D, R = dt_projs_weight.shape
    L = H * W

    if nrows < 1:
        nrows = 2 if D % 2 == 0 else 1

    xs = BiScan2D.apply(x)
    x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
    if x_proj_bias is not None:
        x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
    dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
    dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)
    xs = xs.view(B, -1, L).to(torch.float)
    dts = dts.contiguous().view(B, -1, L).to(torch.float)
    As = -torch.exp(A_logs.to(torch.float))
    Bs = Bs.contiguous().to(torch.float)
    Cs = Cs.contiguous().to(torch.float)
    Ds = Ds.to(torch.float)
    delta_bias = dt_projs_bias.view(-1).to(torch.float)

    ys = SelectiveScan.apply(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows).view(B, K, -1, H, W)

    y = BiMerge2D.apply(ys)
    y = y.permute(0, 2, 3, 1).contiguous()
    if out_norm is not None:
        y = out_norm(y)

    return y.to(x.dtype) if to_dtype else y

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.activ = nn.Sigmoid()
        self.e_lambda = e_lambda

    def forward(self, x):

        b, c, h, w = x.size()
        n = h * w - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.activ(y)


class ECAAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = self.gap(x).squeeze(-1).transpose(-1, -2)
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)
        return x * self.act(y)


class ECAAttention1D(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        y = x.unsqueeze(1)
        y = self.conv(y)
        y = self.act(y).squeeze(1)
        return x * y


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


class HybricMamba(nn.Module):
    def __init__(self, dims=(3, 16, 32, 56, 96), num_classes=43,
                 mbconv_expand_ratio=4, ssm_d_state=8, ssm_ratio=1.0,
                 drop_path_rate=0.10, classifier_dropout=0.2,
                 stem_width=24, cnn_blocks=(1, 1), mamba_blocks=(1, 1),
                 use_aux=True, ssm_frac=0.5, conv_frac=0.3):
        super().__init__()
        self.use_aux = use_aux
        dpr = torch.linspace(0, drop_path_rate, 4).tolist()

        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_width, 3, padding=1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.SiLU(inplace=True),
            nn.Conv2d(stem_width, dims[0], 1, bias=False),
            nn.BatchNorm2d(dims[0]),
            SimAM()
        )

        self.stage1_cnn = CNNStage(dims[0], dims[1], mbconv_expand_ratio, dpr[0], cnn_blocks[0])
        self.stage2_mamba = MambaStage(dims[1], dims[2], dpr[1], mamba_blocks[0], ssm_d_state, ssm_ratio,
                                       ssm_frac=ssm_frac, conv_frac=conv_frac)
        self.stage3_cnn = CNNStage(dims[2], dims[3], mbconv_expand_ratio, dpr[2], cnn_blocks[1])
        self.stage4_mamba = MambaStage(dims[3], dims[4], dpr[3], mamba_blocks[1], ssm_d_state, ssm_ratio,
                                       ssm_frac=ssm_frac, conv_frac=conv_frac)

        # Dual Pooling Head (Avg + Max) Multi-stage Classifier
        concat_dim = (dims[2] + dims[3] + dims[4]) * 2
        self.head_norm = nn.LayerNorm(concat_dim, eps=1e-6)
        self.head_gate = ECAAttention1D(kernel_size=3)
        self.dropout = nn.Dropout(classifier_dropout)
        self.head = nn.Linear(concat_dim, num_classes)

        if self.use_aux:
            self.aux_head2 = nn.Linear(dims[2] * 2, num_classes)
            self.aux_head3 = nn.Linear(dims[3] * 2, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.)

    def forward_features(self, x):
        x = self.stem(x)
        x1 = self.stage1_cnn(x)
        x2 = self.stage2_mamba(x1.permute(0, 2, 3, 1))
        x3 = self.stage3_cnn(x2.permute(0, 3, 1, 2))
        x4 = self.stage4_mamba(x3.permute(0, 2, 3, 1))
        return x2, x3, x4

    def forward(self, x):
        x2, x3, x4 = self.forward_features(x)

        # Spatial-to-Vector Dual Pooling
        f2 = x2.permute(0, 3, 1, 2)
        f4 = x4.permute(0, 3, 1, 2)

        p2_avg = F.adaptive_avg_pool2d(f2, 1).flatten(1)
        p2_max = F.adaptive_max_pool2d(f2, 1).flatten(1)

        p3_avg = F.adaptive_avg_pool2d(x3, 1).flatten(1)
        p3_max = F.adaptive_max_pool2d(x3, 1).flatten(1)

        p4_avg = F.adaptive_avg_pool2d(f4, 1).flatten(1)
        p4_max = F.adaptive_max_pool2d(f4, 1).flatten(1)

        p2 = torch.cat([p2_avg, p2_max], dim=1)
        p3 = torch.cat([p3_avg, p3_max], dim=1)
        p4 = torch.cat([p4_avg, p4_max], dim=1)

        feat = torch.cat([p2, p3, p4], dim=1)
        feat = self.head_gate(self.head_norm(feat))
        logits = self.head(self.dropout(feat))

        if self.training and self.use_aux:
            aux2 = self.aux_head2(p2)
            aux3 = self.aux_head3(p3)
            return logits, aux2, aux3

        return logits


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == "__main__":
    model = HybricMamba(
        dims=(3, 16, 32, 56, 96),
        num_classes=43,
        mbconv_expand_ratio=4,
        ssm_d_state=8,
        mamba_blocks=(1, 1),
        ssm_frac=0.5,
        conv_frac=0.3,
        use_aux=True,
    )

    total, trainable = count_parameters(model)
    print(f"--> Tong tham so: {total:,}")
    print(f"--> Tham so co the huan luyen: {trainable:,}")

    x = torch.randn(2, 3, 32, 32)
    model.train()
    out, aux2, aux3 = model(x)
    print("Train Output Shapes:", out.shape, aux2.shape, aux3.shape)

    model.eval()
    out = model(x)
    print("Eval Output Shape:", out.shape)