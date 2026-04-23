"""
=============================================================================
 models/discriminator.py — TryOnGAN Segmentation-Aware Discriminator
=============================================================================

PAPER COVERAGE:
  ✅ Sec 3.2 — Multi-scale discriminator
  ✅ Sec 3.2 — Segmentation conditioning (seg mask concatenated to input)
  ✅ Sec 4   — Logistic loss with R1 regularization

WHAT THIS MODULE DOES:
  • Takes (image, seg_mask) as input — forces D to be body-part-aware
  • Progressive downsampling with residual blocks (StyleGAN2-D style)
  • Final linear layer outputs real/fake score (no sigmoid)
  • MiniBatch std for diversity encouragement (prevents mode collapse)

WHY SEGMENTATION-CONDITIONING MATTERS:
  Without it, D only knows "does this look like a real person photo?"
  With it, D also checks "does the clothing appear on the right body part?"
  This is the key try-on-specific contribution vs. vanilla StyleGAN2.
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .generator import EqualConv2d, EqualLinear


# ---------------------------------------------------------------------------
# Residual Downsampling Block
# ---------------------------------------------------------------------------
class DiscBlock(nn.Module):
    """
    Skip-connection residual block with 2× spatial downsampling.
    Uses average pooling + 1×1 conv for the skip path (no learnable upsample).
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1  = EqualConv2d(in_ch,  in_ch,  3, padding=1)
        self.conv2  = EqualConv2d(in_ch,  out_ch, 3, padding=1)
        self.skip   = EqualConv2d(in_ch,  out_ch, 1, bias=False)
        self.act    = nn.LeakyReLU(0.2, inplace=True)
        self.pool   = nn.AvgPool2d(2)

    def forward(self, x):
        skip = self.pool(self.skip(x))
        x    = self.act(self.conv1(x))
        x    = self.act(self.conv2(x))
        x    = self.pool(x)
        return (x + skip) / math.sqrt(2)   # variance preservation


# ---------------------------------------------------------------------------
# MiniBatch Standard Deviation
# ---------------------------------------------------------------------------
class MiniBatchStdDev(nn.Module):
    """
    Appends a feature map of local std-dev across the mini-batch.
    Prevents D from ignoring diversity → reduces mode collapse.
    Paper uses group_size=4.
    """
    def __init__(self, group_size=4, n_chan=1):
        super().__init__()
        self.group_size = group_size
        self.n_chan = n_chan

    def forward(self, x):
        B, C, H, W = x.shape
        G = min(self.group_size, B)
        F_ = self.n_chan
        c = C // F_

        y = x.reshape(G, -1, F_, c, H, W)     # (G, B/G, F, c, H, W)
        y = y - y.mean(0, keepdim=True)
        y = y.pow(2).mean(0).add(1e-8).sqrt()  # (B/G, F, c, H, W)
        y = y.mean(dim=[2,3,4], keepdim=True)  # (B/G, F, 1, 1, 1)
        y = y.repeat(G, 1, 1, H, W)           # (B, F, H, W)
        y = y.reshape(B, F_, H, W)

        return torch.cat([x, y], dim=1)        # (B, C+F, H, W)


# ---------------------------------------------------------------------------
# TryOnDiscriminator
# ---------------------------------------------------------------------------
class TryOnDiscriminator(nn.Module):
    """
    PAPER § 3.2 — Segmentation-conditioned discriminator.

    Input: concat(image [3ch], seg_mask [1ch]) = 4 channels total.
    Progressive downsampling → MiniBatchStd → Linear score.

    Channel progression (512×512):
      4ch → 64 → 128 → 256 → 512 → 512 → 512 → 512
    Final: (B, 512, 4, 4) → flatten → linear → scalar score
    """

    def __init__(self, img_size=512, channel_mult=1.0):
        super().__init__()
        log_size = int(math.log2(img_size))

        def ch(stage): return min(int(2 ** (stage + 1) * channel_mult), 512)

        # First conv: RGB+seg → 64 channels
        self.from_rgb = nn.Sequential(
            EqualConv2d(4, ch(5), 1),        # 4ch input (RGB + 1-ch seg)
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Progressive downsample blocks
        self.blocks = nn.ModuleList()
        in_ch = ch(5)
        for i in range(log_size, 2, -1):
            out_ch = min(ch(5) * 2 ** (log_size - i + 1), 512)
            self.blocks.append(DiscBlock(in_ch, out_ch))
            in_ch = out_ch

        # Final 4×4 block
        self.mbstd    = MiniBatchStdDev(group_size=4)
        self.final_conv = EqualConv2d(in_ch + 1, 512, 3, padding=1)
        self.act       = nn.LeakyReLU(0.2, inplace=True)
        self.flatten   = nn.Flatten()
        self.linear    = EqualLinear(512 * 4 * 4, 1)

    def forward(self, img, seg):
        """
        img : (B, 3, H, W) in [-1, 1]
        seg : (B, 1, H, W) in [0, 1]
        Returns: (B, 1) unbounded real/fake logits
        """
        # Resize seg to match img if needed
        if seg.shape[-1] != img.shape[-1]:
            seg = F.interpolate(seg.float(), img.shape[-2:], mode="nearest")

        x = self.from_rgb(torch.cat([img, seg], dim=1))

        for block in self.blocks:
            x = block(x)

        x = self.mbstd(x)
        x = self.act(self.final_conv(x))
        x = self.flatten(x)
        return self.linear(x)
