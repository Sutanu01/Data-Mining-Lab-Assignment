"""
=============================================================================
 models/encoder.py — Pose & Cloth Encoders
=============================================================================

PAPER COVERAGE:
  ✅ Sec 3.1 — Pose conditioning via heatmap encoding into style space
  ✅ Sec 3.1 — Cloth texture encoding into style space
  ✅ Sec 3.3 — Shared encoder used for latent optimization at inference

WHAT THIS MODULE DOES:
  PoseEncoder   : 18-channel heatmap (one per keypoint) → style vector (B, D)
  ClothEncoder  : RGB cloth image → style vector (B, D)

  Both encoders output in the W space so they can be directly injected
  into StyleGAN2's mapping network via concatenation.

WHY SEPARATE ENCODERS?
  Pose and cloth are disentangled by design:
  - Pose controls body shape/layout
  - Cloth controls texture appearance
  Separating them lets us swap cloth at inference without changing pose.

ARCHITECTURE CHOICE:
  Lightweight ResNet-style with global average pooling.
  Kept small (4 residual blocks) to avoid bottlenecking training speed.
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared Backbone
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(8, ch),
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.net(x))


class DownBlock(nn.Module):
    """Halve spatial, double channels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(8, out_ch)
        self.act  = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class LightEncoder(nn.Module):
    """
    Generic lightweight encoder: spatial image → style vector.
    Input:  (B, in_ch, H, W)
    Output: (B, out_dim)
    """
    def __init__(self, in_ch, out_dim):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
        )
        # 512→256→128→64→32 (5 downsamples)
        self.downs = nn.ModuleList([
            DownBlock(32,  64),
            DownBlock(64,  128),
            DownBlock(128, 256),
            DownBlock(256, 256),
            DownBlock(256, 256),
        ])
        self.res_blocks = nn.ModuleList([ResBlock(256) for _ in range(3)])
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.head   = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        for d in self.downs:
            x = d(x)
        for r in self.res_blocks:
            x = r(x)
        x = self.pool(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# PoseEncoder  (paper § 3.1)
# ---------------------------------------------------------------------------
class PoseEncoder(LightEncoder):
    """
    Encodes 18-channel body-pose heatmaps into a style vector.
    Each channel = gaussian blob around one keypoint (OpenPose format).
    The encoder learns which poses → which W-space regions.
    """
    def __init__(self, out_dim=512):
        super().__init__(in_ch=18, out_dim=out_dim)


# ---------------------------------------------------------------------------
# ClothEncoder  (paper § 3.1)
# ---------------------------------------------------------------------------
class ClothEncoder(LightEncoder):
    """
    Encodes a cloth image (RGB) into a style vector.
    Used to condition the generator on the garment texture.
    At inference: swap cloth_enc(new_cloth) to try on different garments.
    """
    def __init__(self, out_dim=512):
        super().__init__(in_ch=3, out_dim=out_dim)
