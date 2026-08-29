"""
Module 6 - DenseNet-121 Backbone (Algorithm 4)
==================================================
Implements the dense-connectivity principle behind DenseNet-121 -- every
layer within a block receives the concatenated feature maps of every
preceding layer in that block, then a 1x1 transition layer compresses
channels between blocks -- as a compact 2D network sized to train in
reasonable time on CPU rather than the full ~8M-parameter torchvision
DenseNet121 (timed at ~3.4s/iteration on this sandbox's CPU, which made a
12-epoch run take ~15 minutes for this module alone).

SCOPE NOTE (same substitution logic as cnn2d.py): 2D, not 3D, because the
dataset is 2D slices; compact rather than full-depth DenseNet-121 because
this increment runs on CPU with no GPU. The dense-block / transition-layer
/ global-pool / dropout-head structure is otherwise the genuine DenseNet
design (concatenated feature reuse across every layer in a block, channel
compression between blocks), not a renamed CNN2D. Growth rate, block
count, and layers-per-block are the parameters to scale up first if this
moves to GPU or the full torchvision densenet121 topology is wanted.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DenseLayer(nn.Module):
    """BN -> ReLU -> Conv(3x3, growth_rate), output concatenated with input."""

    def __init__(self, in_channels: int, growth_rate: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv(self.relu(self.bn(x)))
        return torch.cat([x, out], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, n_layers: int):
        super().__init__()
        layers = []
        ch = in_channels
        for _ in range(n_layers):
            layers.append(DenseLayer(ch, growth_rate))
            ch += growth_rate
        self.block = nn.Sequential(*layers)
        self.out_channels = ch

    def forward(self, x):
        return self.block(x)


class TransitionLayer(nn.Module):
    """1x1 conv to halve channels, followed by 2x2 average pooling."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.pool(self.conv(self.relu(self.bn(x))))


class DenseNet2D(nn.Module):
    """Compact DenseNet standing in for Algorithm 4's DenseNet-121 (3D).
    Features a `self.features` attribute (dense blocks + transitions, pre
    global-pool) so visualize_pipeline.py's Grad-CAM hook works against it
    exactly like it does against CNN2D."""

    def __init__(self, num_classes: int = 3, dropout_p: float = 0.35,
                 in_channels: int = 1, growth_rate: int = 8,
                 block_layers=(3, 3, 3)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        ch = 16
        blocks = []
        for i, n_layers in enumerate(block_layers):
            block = DenseBlock(ch, growth_rate, n_layers)
            blocks.append(block)
            ch = block.out_channels
            if i != len(block_layers) - 1:
                out_ch = ch // 2
                blocks.append(TransitionLayer(ch, out_ch))
                ch = out_ch
        blocks.append(nn.BatchNorm2d(ch))
        blocks.append(nn.ReLU(inplace=True))
        self.features = nn.Sequential(*blocks)   # dense blocks + transitions + final BN/ReLU
        self.out_channels = ch

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)   # MC Dropout target layer
        self.classifier = nn.Linear(ch, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = DenseNet2D(num_classes=3)
    dummy = torch.randn(4, 1, 128, 128)
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}  (expected: (4, 3))")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
