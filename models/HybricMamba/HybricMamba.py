import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_

from .layers.attention import SimAM, ECAAttention1D
from .stage.CNN_Stage import CNNStage
from .stage.Mamba_Stage import MambaStage

try:
    SSMODE = "sscore"
    import selective_scan_cuda_core
except Exception:
    SSMODE = "mamba_ssm"
    try:
        import selective_scan_cuda
    except Exception:
        selective_scan_cuda = None


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