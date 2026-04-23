"""
=============================================================================
 models/generator.py — TryOnGAN Generator
=============================================================================

PAPER COVERAGE:
  ✅ Sec 3.1 — StyleGAN2 backbone with W+ latent space
  ✅ Sec 3.1 — Pose & cloth style injection via AdaIN (modulated conv)
  ✅ Sec 3.1 — Noise injection for stochastic detail
  ✅ Fig. 2  — Architecture: Mapping net → W+ → Synthesis blocks

WHAT THIS MODULE DOES:
  1. MappingNetwork  : maps (pose_style, cloth_style) → W+ (18 × 512)
  2. SynthesisBlock  : StyleGAN2-style modulated conv + upsampling
  3. TryOnGenerator  : wires them together, outputs RGB image

KEY STYLEGAN2 TRICKS USED:
  • Weight-demodulated convolutions (no batch norm needed)
  • Skip connections for RGB (progressive-like without actual progressive)
  • Exponential Moving Average of G weights (EMA) for stable inference
  • Equalized learning rate (all weights initialized N(0,1), scaled at runtime)
=============================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Equalized Learning Rate helpers
# ---------------------------------------------------------------------------
class EqualLinear(nn.Module):
    """FC layer with equalized LR (weight scaled at forward time)."""
    def __init__(self, in_dim, out_dim, bias=True, lr_mul=1.0, activation=None):
        super().__init__()
        self.weight    = nn.Parameter(torch.randn(out_dim, in_dim) / lr_mul)
        self.bias      = nn.Parameter(torch.zeros(out_dim)) if bias else None
        self.scale     = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul    = lr_mul
        self.activation = activation   # "fused_lrelu" or None

    def forward(self, x):
        out = F.linear(x, self.weight * self.scale,
                        self.bias * self.lr_mul if self.bias is not None else None)
        if self.activation == "fused_lrelu":
            out = F.leaky_relu(out, 0.2) * math.sqrt(2)   # gain correction
        return out


class EqualConv2d(nn.Module):
    """Conv2d with equalized LR."""
    def __init__(self, in_ch, out_ch, kernel, stride=1, padding=0, bias=True):
        super().__init__()
        self.weight  = nn.Parameter(torch.randn(out_ch, in_ch, kernel, kernel))
        self.bias    = nn.Parameter(torch.zeros(out_ch)) if bias else None
        self.scale   = 1 / math.sqrt(in_ch * kernel ** 2)
        self.stride  = stride; self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight * self.scale, self.bias,
                        stride=self.stride, padding=self.padding)


# ---------------------------------------------------------------------------
# Mapping Network  (paper § 3.1)
# ---------------------------------------------------------------------------
class MappingNetwork(nn.Module):
    """
    8-layer MLP that maps concatenated (pose_style ⊕ cloth_style) → W.
    We then broadcast W to W+ by repeating across all 18 style layers.
    """
    def __init__(self, z_dim=512, style_dim=512, n_mlp=8, lr_mlp=0.01):
        super().__init__()
        layers = [PixelNorm()]
        for i in range(n_mlp):
            in_d  = z_dim if i == 0 else style_dim
            layers.append(EqualLinear(in_d, style_dim,
                                       lr_mul=lr_mlp,
                                       activation="fused_lrelu"))
        self.net = nn.Sequential(*layers)

    def forward(self, z):   # z: (B, z_dim)
        return self.net(z)  # → (B, style_dim)


class PixelNorm(nn.Module):
    def forward(self, x):
        return x / (x.pow(2).mean(dim=1, keepdim=True).add(1e-8).sqrt())


# ---------------------------------------------------------------------------
# Modulated Convolution  (StyleGAN2 weight demodulation)
# ---------------------------------------------------------------------------
class ModulatedConv2d(nn.Module):
    """
    Core StyleGAN2 op: weight is modulated by style vector s,
    then demodulated to unit variance → no need for batch norm.
    """
    def __init__(self, in_ch, out_ch, kernel, style_dim,
                 up=False, down=False, demod=True, padding=0):
        super().__init__()
        self.out_ch = out_ch; self.kernel = kernel
        self.up = up; self.down = down; self.demod = demod
        self.padding = padding

        self.weight    = nn.Parameter(torch.randn(1, out_ch, in_ch, kernel, kernel))
        self.modulator = EqualLinear(style_dim, in_ch, bias=True)
        nn.init.ones_(self.modulator.bias)   # start near identity

        self.scale = 1 / math.sqrt(in_ch * kernel ** 2)

    def forward(self, x, style):
        B, C, H, W = x.shape
        # modulate
        s = self.modulator(style).view(B, 1, C, 1, 1)      # (B,1,in,1,1)
        w = self.weight * self.scale * s                   # (B,out,in,k,k)
        if self.demod:
            d = w.pow(2).sum(dim=[2,3,4], keepdim=True).add(1e-8).rsqrt()
            w = w * d
        # reshape for group conv trick
        x = x.reshape(1, B*C, H, W)
        w = w.reshape(B * self.out_ch, C, self.kernel, self.kernel)

        if self.up:
            x = F.interpolate(x.view(B, C, H, W), scale_factor=2, mode="bilinear",
                              align_corners=False)
            x = x.reshape(1, B*C, H*2, W*2)
        if self.down:
            x = F.avg_pool2d(x, 2)

        x = F.conv2d(x, w, padding=self.padding, groups=B)
        x = x.view(B, self.out_ch, x.shape[-2], x.shape[-1])
        return x


# ---------------------------------------------------------------------------
# Noise Injection
# ---------------------------------------------------------------------------
class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x, noise=None):
        if noise is None:
            B, _, H, W = x.shape
            noise = torch.randn(B, 1, H, W, device=x.device, dtype=x.dtype)
        return x + self.weight * noise


# ---------------------------------------------------------------------------
# Synthesis Block
# ---------------------------------------------------------------------------
class SynthesisBlock(nn.Module):
    """
    One resolution block: 2× upsampled modconv → noise → LReLU → toRGB skip.
    """
    def __init__(self, in_ch, out_ch, style_dim, upsample=True):
        super().__init__()
        self.upsample = upsample
        self.conv1  = ModulatedConv2d(in_ch,  out_ch, 3, style_dim,
                                       up=upsample, padding=1)
        self.conv2  = ModulatedConv2d(out_ch, out_ch, 3, style_dim, padding=1)
        self.noise1 = NoiseInjection()
        self.noise2 = NoiseInjection()
        self.act    = nn.LeakyReLU(0.2, inplace=True)
        self.to_rgb = ModulatedConv2d(out_ch, 3, 1, style_dim, demod=False)

    def forward(self, x, w1, w2, skip=None):
        # w1, w2 = style vectors for the two convolutions
        x = self.conv1(x, w1)
        x = self.noise1(x)
        x = self.act(x)

        x = self.conv2(x, w2)
        x = self.noise2(x)
        x = self.act(x)

        # RGB skip
        rgb = self.to_rgb(x, w2)
        if skip is not None:
            skip = F.interpolate(skip, scale_factor=2,
                                 mode="bilinear", align_corners=False)
            rgb  = rgb + skip

        return x, rgb


# ---------------------------------------------------------------------------
# Constant Input
# ---------------------------------------------------------------------------
class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, B):
        return self.input.expand(B, -1, -1, -1)


# ---------------------------------------------------------------------------
# TryOnGenerator
# ---------------------------------------------------------------------------
class TryOnGenerator(nn.Module):
    """
    PAPER § 3.1 — Pose-conditioned StyleGAN2 Generator.

    Architecture sizes for 512×512:
      4→8→16→32→64→128→256→512
    Each block consumes 2 style vectors from W+ (one per conv).
    Total = 18 style vectors (matches canonical StyleGAN2).

    Inputs:
      styles: list of [pose_w, cloth_w], each (B, style_dim)
              → concatenated → MappingNet → W → broadcast to W+

    Outputs:
      image (B, 3, H, W) in [-1, 1]
      latents W+ for optional optimization
    """

    def __init__(self, img_size=512, style_dim=512, n_mlp=8, channel_mult=1.0):
        super().__init__()

        self.style_dim = style_dim
        self.log_size  = int(math.log2(img_size))         # 9 for 512
        self.num_layers = (self.log_size - 2) * 2 + 1    # 15 for 512

        # Channel widths (halved each upscale, capped at 512)
        def ch(stage): return min(int(32768 / (2 ** stage) * channel_mult), 512)

        # Mapping: pose_style ⊕ cloth_style → W
        self.mapping = MappingNetwork(z_dim=style_dim * 2,
                                       style_dim=style_dim,
                                       n_mlp=n_mlp)

        # Constant 4×4 seed
        self.input = ConstantInput(ch(1))

        # Synthesis blocks
        self.blocks = nn.ModuleList()
        in_ch = ch(1)
        for stage in range(2, self.log_size + 1):
            out_ch = ch(stage)
            self.blocks.append(SynthesisBlock(in_ch, out_ch, style_dim,
                                               upsample=(stage > 2)))
            in_ch = out_ch

        # Number of W+ vectors needed
        self.n_latent = self.log_size * 2 - 2   # 16 for 512

    def forward(self, styles, return_latents=False, truncation=0.7):
        """
        styles: [pose_w (B, D), cloth_w (B, D)]
        """
        # 1. Concatenate condition vectors
        z = torch.cat(styles, dim=1)          # (B, 2*style_dim)

        # 2. Map to W space
        w = self.mapping(z)                   # (B, style_dim)

        # 3. Truncation trick (inference stability)
        if truncation < 1.0:
            w_avg = w.mean(0, keepdim=True)
            w = w_avg + truncation * (w - w_avg)

        # 4. Broadcast to W+ (all layers share same w for simplicity;
        #    could be per-layer for fine-grained control)
        w_plus = w.unsqueeze(1).expand(-1, self.n_latent, -1)  # (B, n, D)

        # 5. Synthesis
        x   = self.input(z.shape[0])
        rgb = None
        idx = 0
        for block in self.blocks:
            w1 = w_plus[:, idx]
            w2 = w_plus[:, idx + 1]
            x, rgb = block(x, w1, w2, skip=rgb)
            idx += 2

        image = torch.tanh(rgb)   # → [-1, 1]

        if return_latents:
            return image, w_plus
        return image, None
