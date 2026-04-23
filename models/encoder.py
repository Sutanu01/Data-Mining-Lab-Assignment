"""
=============================================================================
 models/encoder.py — Pose & Cloth Encoders
=============================================================================

PAPER COVERAGE:
  ✅ Sec 3.1 — Pose conditioning via heatmap encoding into style space
  ✅ Sec 3.1 — Cloth texture encoding into style space
  ✅ Sec 3.3 — Shared encoder used for latent optimization at inference

WHAT THIS MODULE DOES:
  PoseEncoder   : 18-channel heatmap (one per keypoint) → style vector
                  Output: (B, style_dim)  OR  (B, n_latent, style_dim) in W+
  ClothEncoder  : RGB cloth image → style vector with multi-scale texture
                  Output: (B, style_dim)  OR  (B, n_latent, style_dim) in W+

  Both encoders output in the W space so they can be directly injected
  into StyleGAN2's mapping network via concatenation (default) OR broadcast
  to W+ for per-layer fine-grained style control (pass n_latent > 0).

WHY SEPARATE ENCODERS?
  Pose and cloth are disentangled by design:
  - Pose controls body shape/layout        → sparse 18-channel heatmap input
  - Cloth controls texture appearance      → dense RGB input, multi-scale
  Separating them lets us swap cloth at inference without changing pose.

ARCHITECTURE:
  Both share a LightEncoder backbone (ResNet-style, GlobalAvgPool → MLP).
  ClothEncoder adds a multi-scale texture aggregation path (FPN-lite) to
  preserve fine garment detail before pooling.

  Channel schedule (512×512 input, 5 downsamples → 16×16 feature map):
    stem: in_ch → 32
    down: 32→64→128→256→256→256
    res:  256 × 3 blocks
    pool: AdaptiveAvgPool(1) → (B, 256, 1, 1)
    head: Linear(256, style_dim) → PixelNorm

SYNC POINTS WITH REST OF CODEBASE:
  • train.py        : pose_enc(pose_hmap) and cloth_enc(cloth_img) both
                      return (B, style_dim); passed as [w_pose, w_cloth] to G
  • generator.py    : G([w_pose, w_cloth]) cats them → (B, 2*style_dim)
                      → MappingNetwork → W → broadcast to W+
  • inference.py    : encoders called with requires_grad_() for latent opt;
                      freeze()/unfreeze() used around optimization loop
  • losses.py       : no direct dependency; perceptual loss applied to G output
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num_groups(channels: int, target: int = 8) -> int:
    """
    Return the largest divisor of `channels` that is <= target.
    Prevents GroupNorm from failing when channels < target groups.
    """
    for g in range(min(target, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1                 # fallback: LayerNorm-like (1 group)


class PixelNorm(nn.Module):
    """
    Normalise each vector to unit length along the channel dimension.
    Used to keep encoder outputs in the same distribution as the W space
    expected by the StyleGAN2 MappingNetwork (which also starts with PixelNorm).
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / (x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """
    Pre-activation residual block.
    GroupNorm groups adapt automatically to channel count.
    """
    def __init__(self, ch: int):
        super().__init__()
        g = _num_groups(ch)
        self.net = nn.Sequential(
            nn.GroupNorm(g, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(g, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DownBlock(nn.Module):
    """
    Halve spatial resolution, optionally change channel count.
    Strided conv → GroupNorm → SiLU.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        g = _num_groups(out_ch)
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(g, out_ch)
        self.act  = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


# ---------------------------------------------------------------------------
# Lite FPN neck  (used by ClothEncoder only)
# ---------------------------------------------------------------------------

class LiteFPN(nn.Module):
    """
    Lightweight top-down Feature Pyramid that fuses multi-scale cloth
    features before global pooling.  Produces a single (B, out_ch, 1, 1)
    tensor via AdaptiveAvgPool.

    Why needed for cloth but not pose?
    Pose heatmaps are sparse and low-frequency; a single global vector
    captures them well.  Garment textures are high-frequency and spatially
    dense, so pooling too early loses detail.  The FPN neck aggregates
    features at 3 scales (P3/P4/P5) by projecting them to the same width
    and summing before the final pool, preserving both coarse silhouette
    and fine texture information.

    in_channels : list of channel widths for [p3, p4, p5] feature maps
    out_ch      : unified projection width
    """
    def __init__(self, in_channels: list, out_ch: int = 256):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Conv2d(c, out_ch, 1, bias=False) for c in in_channels
        ])
        self.pool  = nn.AdaptiveAvgPool2d(1)

    def forward(self, features: list) -> torch.Tensor:
        """
        features : [p3 (B,C3,H3,W3), p4 (B,C4,H4,W4), p5 (B,C5,H5,W5)]
        Upsample smaller maps to match p3's spatial size, sum, then pool.
        """
        target_size = features[0].shape[-2:]
        out = None
        for proj, feat in zip(self.projs, features):
            f = proj(feat)
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size,
                                  mode="bilinear", align_corners=False)
            out = f if out is None else out + f
        return self.pool(out)       # (B, out_ch, 1, 1)


# ---------------------------------------------------------------------------
# LightEncoder  — shared backbone
# ---------------------------------------------------------------------------

class LightEncoder(nn.Module):
    """
    Generic lightweight encoder: spatial image → style vector.

    Input  : (B, in_ch, H, W)         — H=W=img_size (default 512)
    Output : (B, style_dim)            — W-space vector, pixel-normalised
          OR (B, n_latent, style_dim)  — W+ tensor (if n_latent > 0)

    Channel schedule (5 downs, 512 → 16px feature map):
      32 → 64 → 128 → 256 → 256 → 256
    After 3 ResBlocks + GlobalAvgPool → Linear(256, style_dim) → PixelNorm.

    n_latent=0  : single W vector (default, matches train.py / inference.py)
    n_latent>0  : W+ broadcast; used when the caller wants per-layer control
                  (generator.py currently broadcasts internally, but this lets
                  us expose it at the encoder level for future fine-tuning)
    """

    # Channel schedule: index = number of downs applied so far
    _CHANNELS = [32, 64, 128, 256, 256, 256]

    def __init__(self, in_ch: int, style_dim: int = 512, n_latent: int = 0):
        super().__init__()
        self.style_dim = style_dim
        self.n_latent  = n_latent       # 0 = return flat W; >0 = return W+

        # ── Stem ────────────────────────────────────────────────────────────
        stem_ch = self._CHANNELS[0]     # 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_ch, 3, padding=1, bias=False),
            nn.GroupNorm(_num_groups(stem_ch), stem_ch),
            nn.SiLU(),
        )

        # ── Downsample blocks (512→256→128→64→32→16) ───────────────────────
        self.downs = nn.ModuleList()
        for i in range(len(self._CHANNELS) - 1):
            self.downs.append(
                DownBlock(self._CHANNELS[i], self._CHANNELS[i + 1])
            )

        # ── Residual refinement at lowest resolution ─────────────────────
        self.res_blocks = nn.ModuleList(
            [ResBlock(self._CHANNELS[-1]) for _ in range(4)]
        )

        # ── Aggregation & projection ─────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),                                   # (B, 256)
            nn.Linear(self._CHANNELS[-1], style_dim),
            nn.LayerNorm(style_dim),
        )
        self.pixel_norm = PixelNorm()

        self._init_weights()

    # ── Weight initialisation ────────────────────────────────────────────
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Core forward ─────────────────────────────────────────────────────
    def _encode(self, x: torch.Tensor):
        """Return intermediate feature maps and the pooled feature."""
        x = self.stem(x)
        features = []
        for down in self.downs:
            x = down(x)
            features.append(x)
        for res in self.res_blocks:
            x = res(x)
        pooled = self.pool(x)           # (B, 256, 1, 1)
        return features, pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, in_ch, H, W)
        Returns:
          (B, style_dim)              if self.n_latent == 0
          (B, n_latent, style_dim)    if self.n_latent >  0
        """
        _, pooled = self._encode(x)
        w = self.pixel_norm(self.head(pooled))      # (B, style_dim)

        if self.n_latent > 0:
            # Broadcast single W → W+  (per-layer identical initialisation;
            # can be fine-tuned per-layer during latent optimisation)
            w = w.unsqueeze(1).expand(-1, self.n_latent, -1)  # (B, L, D)
        return w

    # ── Utility: freeze / unfreeze for inference latent opt ──────────────
    def freeze(self):
        """Freeze all encoder parameters (used before latent optimisation)."""
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def unfreeze(self):
        """Restore gradient flow (used when fine-tuning encoders end-to-end)."""
        for p in self.parameters():
            p.requires_grad_(True)
        return self


# ---------------------------------------------------------------------------
# PoseEncoder  (paper § 3.1)
# ---------------------------------------------------------------------------

class PoseEncoder(LightEncoder):
    """
    Encodes 18-channel body-pose heatmaps into a W-space style vector.

    Input  : (B, 18, H, W)  — one Gaussian blob per OpenPose joint
    Output : (B, style_dim) — pixel-normalised W vector

    Design rationale:
    Pose heatmaps are sparse (most pixels near zero) and low-frequency, so
    a straightforward downsample + pool architecture is sufficient.  No FPN
    neck is needed; a single global average pool captures joint layout well.

    Sync with generator.py:
      The output is passed as styles[0] → G([w_pose, w_cloth]).
      G cats it with w_cloth → (B, 2*style_dim) → MappingNetwork → W+.
    """
    def __init__(self, out_dim: int = 512, n_latent: int = 0):
        super().__init__(in_ch=18, style_dim=out_dim, n_latent=n_latent)


# ---------------------------------------------------------------------------
# ClothEncoder  (paper § 3.1)
# ---------------------------------------------------------------------------

class ClothEncoder(nn.Module):
    """
    Encodes a cloth/garment RGB image into a W-space style vector.

    Input  : (B, 3, H, W)   — garment image in [-1, 1]
    Output : (B, style_dim) — pixel-normalised W vector

    Design differences from PoseEncoder:
    1. Multi-scale FPN neck aggregates features at P3/P4/P5 (64/32/16 px)
       before global pooling, retaining fine texture information (stripes,
       prints, stitching patterns) that would be lost with early pooling.
    2. Same head + PixelNorm as LightEncoder for W-space compatibility.

    FPN feature widths for default _CHANNELS=[32,64,128,256,256,256]:
      down[2] → 128ch @ H/8   (P3, fine texture)
      down[3] → 256ch @ H/16  (P4, mid-level shape)
      down[4] → 256ch @ H/32  (P5, global silhouette)

    Sync with train.py / inference.py:
      Output used as styles[1] → G([w_pose, w_cloth]).
      At inference: swap cloth_enc(new_cloth) to virtuallychange garments.
    """

    # Indices into the `downs` feature list that form the FPN pyramid
    _FPN_STAGES   = [2, 3, 4]          # P3, P4, P5
    # Matching channel widths from LightEncoder._CHANNELS (after each down)
    _FPN_CHANNELS = [128, 256, 256]

    def __init__(self, out_dim: int = 512, n_latent: int = 0):
        super().__init__()
        self.style_dim = out_dim
        self.n_latent  = n_latent

        # ── Shared backbone ───────────────────────────────────────────────
        self._backbone = LightEncoder(in_ch=3, style_dim=out_dim,
                                      n_latent=0)   # W output handled here

        # ── Multi-scale FPN neck ──────────────────────────────────────────
        self.fpn = LiteFPN(
            in_channels = self._FPN_CHANNELS,
            out_ch      = 256,
        )

        # ── Final projection: fuse backbone + FPN ─────────────────────────
        # backbone head produces (B, out_dim); FPN produces (B, 256, 1, 1)
        # We project FPN → out_dim and add residually.
        self.fpn_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.pixel_norm = PixelNorm()

        self._init_head_weights()

    def _init_head_weights(self):
        for m in self.fpn_proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, 3, H, W) in [-1, 1]
        Returns:
          (B, style_dim)              if self.n_latent == 0
          (B, n_latent, style_dim)    if self.n_latent >  0
        """
        # ── Backbone forward (get both intermediate features and W) ───────
        features, pooled = self._backbone._encode(x)

        # ── Backbone W vector ─────────────────────────────────────────────
        w_backbone = self._backbone.head(pooled)            # (B, style_dim)

        # ── FPN multi-scale aggregation ───────────────────────────────────
        fpn_feats  = [features[i] for i in self._FPN_STAGES]
        fpn_pooled = self.fpn(fpn_feats)                    # (B, 256, 1, 1)
        w_fpn      = self.fpn_proj(fpn_pooled)              # (B, style_dim)

        # ── Fuse: backbone captures global layout; FPN captures texture ───
        # Simple additive fusion (equal weight); could be learned if needed.
        w = self.pixel_norm(w_backbone + w_fpn)             # (B, style_dim)

        if self.n_latent > 0:
            w = w.unsqueeze(1).expand(-1, self.n_latent, -1)
        return w

    # ── Utility: freeze / unfreeze ────────────────────────────────────────
    def freeze(self):
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad_(True)
        return self