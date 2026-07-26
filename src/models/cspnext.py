"""
src/models/cspnext.py
---------------------
Phase 2 - CSPNeXt (the RTMDet backbone) implemented in pure PyTorch.

RTMDet's core architectural contribution is **CSPNeXt**: a CSP-style backbone
built from depthwise 5x5 blocks with SiLU activations and channel attention.
The official implementation lives in mmdetection/mmpretrain, which cannot be
installed in this environment (openmim breaks on Python 3.12, and mmcv has no
wheels for torch 2.12). This module reimplements the same architecture directly
so the contract's "CNN vs RTMDet" benchmark can be run honestly on CPU.

Faithful to the RTMDet paper (Lyu et al., 2022, arXiv:2212.07784):
  * stem   : 3 x 3x3 convs (channel-halved stem)
  * stages : downsample conv -> CSPLayer(CSPNeXtBlock x n)
  * block  : 5x5 depthwise + 1x1 pointwise, residual
  * attn   : channel attention at the end of each CSPLayer
  * act    : SiLU throughout

`CSPNeXtClassifier` wraps the backbone with a global-pool + linear head for the
wheel-count classification task.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvModule(nn.Module):
    """Conv -> BatchNorm -> SiLU (the standard RTMDet conv unit)."""

    def __init__(self, cin: int, cout: int, k: int = 1, s: int = 1,
                 groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, padding=k // 2,
                              groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """Squeeze-excite style attention used at the end of RTMDet CSP layers."""

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, bias=True)
        self.act = nn.Hardsigmoid(inplace=True)

    def forward(self, x):
        return x * self.act(self.fc(self.pool(x)))


class CSPNeXtBlock(nn.Module):
    """
    RTMDet's basic block: a 3x3 conv followed by a **5x5 depthwise-separable**
    conv, with a residual connection. The large depthwise kernel is what gives
    CSPNeXt its wide receptive field at low FLOPs.
    """

    def __init__(self, cin: int, cout: int, expansion: float = 0.5,
                 add_identity: bool = True):
        super().__init__()
        hidden = int(cout * expansion)
        self.conv1 = ConvModule(cin, hidden, k=3, s=1)
        # depthwise 5x5 + pointwise 1x1
        self.dw = ConvModule(hidden, hidden, k=5, s=1, groups=hidden)
        self.pw = ConvModule(hidden, cout, k=1, s=1)
        self.add_identity = add_identity and cin == cout

    def forward(self, x):
        out = self.pw(self.dw(self.conv1(x)))
        return x + out if self.add_identity else out


class CSPLayer(nn.Module):
    """Cross-Stage-Partial layer: split -> n blocks on one branch -> concat."""

    def __init__(self, cin: int, cout: int, num_blocks: int = 1,
                 expand_ratio: float = 0.5, add_identity: bool = True,
                 use_attention: bool = True):
        super().__init__()
        mid = int(cout * expand_ratio)
        self.main_conv = ConvModule(cin, mid, k=1)
        self.short_conv = ConvModule(cin, mid, k=1)
        self.blocks = nn.Sequential(*[
            CSPNeXtBlock(mid, mid, expansion=1.0, add_identity=add_identity)
            for _ in range(num_blocks)
        ])
        self.attention = ChannelAttention(2 * mid) if use_attention else None
        self.final_conv = ConvModule(2 * mid, cout, k=1)

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.blocks(self.main_conv(x))
        out = torch.cat((x_main, x_short), dim=1)
        if self.attention is not None:
            out = self.attention(out)
        return self.final_conv(out)


class SPPFBottleneck(nn.Module):
    """Fast spatial pyramid pooling, as used at the end of the RTMDet backbone."""

    def __init__(self, cin: int, cout: int, k: int = 5):
        super().__init__()
        mid = cin // 2
        self.conv1 = ConvModule(cin, mid, k=1)
        self.pool = nn.MaxPool2d(k, stride=1, padding=k // 2)
        self.conv2 = ConvModule(mid * 4, cout, k=1)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat((x, y1, y2, y3), dim=1))


class CSPNeXt(nn.Module):
    """
    CSPNeXt backbone.

    `widen_factor` / `deepen_factor` follow RTMDet's scaling convention:
        tiny  = (0.375, 0.167)   small = (0.5, 0.33)
        med   = (0.75, 0.67)     large = (1.0, 1.0)
    Defaults here are RTMDet-tiny, which is the right size for CPU training on
    small crop datasets.
    """

    # (out_channels, num_blocks, add_identity, use_spp) per stage @ scale 1.0
    ARCH = [
        (128, 3, True, False),
        (256, 6, True, False),
        (512, 6, True, False),
        (1024, 3, False, True),
    ]

    def __init__(self, widen_factor: float = 0.375,
                 deepen_factor: float = 0.167):
        super().__init__()
        base = int(64 * widen_factor)
        # Stem: three 3x3 convs (RTMDet replaces the single big-stride conv).
        self.stem = nn.Sequential(
            ConvModule(3, base // 2, k=3, s=2),
            ConvModule(base // 2, base // 2, k=3, s=1),
            ConvModule(base // 2, base, k=3, s=1),
        )
        stages = []
        cin = base
        for cout_raw, nblocks_raw, add_id, use_spp in self.ARCH:
            cout = int(cout_raw * widen_factor)
            nblocks = max(round(nblocks_raw * deepen_factor), 1)
            layers = [ConvModule(cin, cout, k=3, s=2)]
            if use_spp:
                layers.append(SPPFBottleneck(cout, cout))
            layers.append(CSPLayer(cout, cout, num_blocks=nblocks,
                                   add_identity=add_id))
            stages.append(nn.Sequential(*layers))
            cin = cout
        self.stages = nn.Sequential(*stages)
        self.out_channels = cin

    def forward(self, x):
        return self.stages(self.stem(x))


class CSPNeXtClassifier(nn.Module):
    """CSPNeXt backbone + global average pool + linear head."""

    def __init__(self, num_classes: int = 4, widen_factor: float = 0.375,
                 deepen_factor: float = 0.167, dropout: float = 0.2):
        super().__init__()
        self.backbone = CSPNeXt(widen_factor, deepen_factor)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.backbone.out_channels, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))


if __name__ == "__main__":
    m = CSPNeXtClassifier(num_classes=4)
    n_params = sum(p.numel() for p in m.parameters())
    x = torch.randn(2, 3, 96, 96)
    y = m(x)
    print(f"CSPNeXt-tiny classifier: {n_params/1e6:.2f}M params, "
          f"out {tuple(y.shape)}")
