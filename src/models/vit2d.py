"""
Module 7 - Vision Transformer Backbone (Algorithm 5)
========================================================
Implements the patch-embedding + positional-encoding + multi-head
self-attention encoder + classification-head structure specified for the
ViT/Swin module, as a compact custom Vision Transformer.

SCOPE NOTE: built from scratch rather than loading torchvision's
`vit_b_16`/Swin because those default to 224x224 3-channel input with
ImageNet-pretrained weights this sandbox cannot download (no network
access to the weights host) and are sized for GPU training. This is a
genuine ViT -- non-overlapping patch embedding via a strided convolution,
a learnable [CLS] token, learnable positional embeddings, and a stack of
standard `nn.TransformerEncoderLayer` blocks (multi-head self-attention +
feed-forward, matching Algorithm 5's diagram) -- just sized for a 64x64
grayscale slice and CPU training.

Dropout is applied to the pooled [CLS] representation before the final
linear head, in the same place as CNN2D/DenseNet2D, so `mc_dropout.py`
needs no changes to work against this backbone.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .. import config


class PatchEmbedding(nn.Module):
    """Non-overlapping patch extraction + linear projection, implemented as
    a strided convolution (Algorithm 5's 'Patch Extraction' + 'Patch
    Embedding' steps combined, as is standard for ViT)."""

    def __init__(self, img_size: int, patch_size: int, in_channels: int, embed_dim: int):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.n_patches = (img_size // patch_size) ** 2

    def forward(self, x):
        x = self.proj(x)                  # (B, embed_dim, H/p, W/p)
        return x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)


class ViT2D(nn.Module):
    """Compact Vision Transformer standing in for Algorithm 5's ViT/Swin."""

    def __init__(self, num_classes: int = 3,
                 img_size: int = config.IMAGE_SIZE[0],
                 patch_size: int = config.VIT_PATCH_SIZE,
                 embed_dim: int = config.VIT_EMBED_DIM,
                 depth: int = config.VIT_DEPTH,
                 num_heads: int = config.VIT_NUM_HEADS,
                 mlp_dim: int = config.VIT_MLP_DIM,
                 dropout_p: float = config.DROPOUT_P,
                 in_channels: int = 1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches

        # [CLS] token (Algorithm 5's implicit classification token) + learnable
        # positional encoding, exactly as named in the module diagram.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)  # Multi-Head Self-Attention + FFN, repeated

        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(p=dropout_p)   # MC Dropout target layer
        self.head = nn.Linear(embed_dim, num_classes)  # Classification Head

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)                        # (B, n_patches, embed_dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos_embed  # prepend [CLS] + add positional encoding
        x = self.encoder(x)                             # Transformer Encoder (repeat x depth)
        x = self.norm(x[:, 0])                          # pooled [CLS] representation
        x = self.dropout(x)
        return self.head(x)                             # Softmax applied by the loss/eval code, not here


if __name__ == "__main__":
    model = ViT2D(num_classes=3)
    dummy = torch.randn(4, 1, *config.IMAGE_SIZE)
    out = model(dummy)
    print(f"Output shape: {tuple(out.shape)}  (expected: (4, 3))")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
