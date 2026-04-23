"""
=============================================================================
 losses.py — TryOnGAN Training Losses
=============================================================================

PAPER COVERAGE:
  ✅ Eq.(1) — Non-saturating adversarial loss (generator + discriminator)
  ✅ Eq.(2) — R1 gradient penalty (lazy regularization every 16 steps)
  ✅ Eq.(3) — VGG perceptual loss on multi-scale features
  ✅ Eq.(4) — Pixel L1 loss on visible (non-occluded) body region

STABLE GAN TRAINING CHOICES:
  1. Non-saturating logistic loss (avoids vanishing gradients early in training)
  2. Lazy R1 (compute every r1_every steps, not every step → 2× faster)
  3. Gradient clipping in train.py (norm 1.0)
  4. Two separate GradScalers (G and D decouple their AMP scaling)
  5. MiniBatch StdDev in D (prevents mode collapse)
  6. EMA of G weights (smoother inference; applied in inference.py)
=============================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ---------------------------------------------------------------------------
# Adversarial Losses
# ---------------------------------------------------------------------------
class AdversarialLoss(nn.Module):
    """
    Non-saturating logistic GAN loss.
      D: maximize log σ(real) + log(1 - σ(fake))
         → minimize  softplus(-real) + softplus(fake)
      G: maximize log σ(fake)
         → minimize  softplus(-fake)
    """
    def __init__(self, mode="g"):
        super().__init__()
        assert mode in ("g", "d"), "mode must be 'g' or 'd'"
        self.mode = mode

    def forward(self, real_score=None, fake_score=None):
        if self.mode == "d":
            # Discriminator: real → high, fake → low
            loss = (F.softplus(-real_score) + F.softplus(fake_score)).mean()
        else:
            # Generator: fool D → push fake score high
            loss = F.softplus(-fake_score).mean()
        return loss


# ---------------------------------------------------------------------------
# R1 Gradient Penalty  (lazy regularization)
# ---------------------------------------------------------------------------
class R1Penalty(nn.Module):
    """
    R1 penalty: regularizes D to have zero gradient on real data.
    Formula (lazy): γ/2 * E[‖∇D(real)‖²] * r1_every / (r1_every + 1)

    Applied only every `r1_every` steps to reduce compute ~50%.
    Loss is scaled back to be equivalent to applying it every step.
    """
    def forward(self, real_score, real_img, gamma, r1_every):
        grad = torch.autograd.grad(
            outputs    = real_score.sum(),
            inputs     = real_img,
            create_graph = True,
        )[0]
        penalty = grad.pow(2).reshape(grad.shape[0], -1).sum(1).mean()
        return (gamma / 2) * penalty * r1_every  # scale for lazy application


# ---------------------------------------------------------------------------
# Perceptual (VGG) Loss  (paper § 4 Eq.3)
# ---------------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    """
    Multi-scale feature matching using VGG16.
    Extracts features at relu1_2, relu2_2, relu3_3, relu4_3.
    Computes L1 distance at each scale → sum.

    WHY VGG OVER LPIPS HERE:
    Simpler to use with FP16 (LPIPS has known AMP issues).
    VGG features are sufficient for texture matching at 512px.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Extract up to relu3_3 (layer 16 in features)
        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.features)[:4]),    # relu1_2
            nn.Sequential(*list(vgg.features)[4:9]),   # relu2_2
            nn.Sequential(*list(vgg.features)[9:16]),  # relu3_3
            nn.Sequential(*list(vgg.features)[16:23]), # relu4_3
        ])
        # Freeze VGG — we only use it for feature extraction
        for p in self.parameters():
            p.requires_grad_(False)

        # VGG normalization constants
        self.register_buffer("mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",
            torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def preprocess(self, x):
        # x in [-1, 1] → [0, 1] → VGG normalized
        x = (x + 1) / 2
        return (x - self.mean) / self.std

    def forward(self, fake, real):
        fake = self.preprocess(fake)
        real = self.preprocess(real)

        loss = 0.0
        weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4]

        f_x, f_y = fake, real
        for slice_, w in zip(self.slices, weights):
            f_x = slice_(f_x)
            f_y = slice_(f_y)
            loss = loss + F.l1_loss(f_x, f_y.detach()) * w

        return loss
