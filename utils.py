"""
=============================================================================
 utils.py — Utility Functions
=============================================================================

PAPER COVERAGE:
  ✅ Sec 5   — Exponential Moving Average of G weights (EMA)
  ✅ Sec 5   — Checkpoint saving / loading

UTILITIES:
  AverageMeter     : running mean for loss logging
  EMA              : tracks shadow weights of generator
  save_checkpoint  : saves G, D, optimizers, step count
  load_checkpoint  : restores from checkpoint
  log_images       : saves training sample grids to disk
  setup_logging    : configures Python logger for rank 0
=============================================================================
"""

import os, logging, math
from pathlib import Path
import torch
import torchvision.utils as vutils


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(output_dir, rank=0):
    handlers = []
    if rank == 0:
        log_path = Path(output_dir) / "train.log"
        handlers.append(logging.FileHandler(log_path))
        handlers.append(logging.StreamHandler())
    logging.basicConfig(
        level   = logging.INFO,
        format  = "%(asctime)s [%(levelname)s] %(message)s",
        handlers = handlers,
    )


# ---------------------------------------------------------------------------
# AverageMeter
# ---------------------------------------------------------------------------
class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / max(self.count, 1)


# ---------------------------------------------------------------------------
# EMA — Exponential Moving Average of Generator Weights
# ---------------------------------------------------------------------------
class EMA:
    """
    Maintains a shadow copy of the generator with exponential moving average.
    EMA weights are used for inference / image logging (smoother output).

    decay=0.9999 matches StyleGAN2 paper.
    """
    def __init__(self, model, decay=0.9999):
        import copy
        self.model  = model
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        self.decay  = decay
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self):
        for (name, ema_p), (_, p) in zip(
            self.shadow.named_parameters(),
            self.model.named_parameters()
        ):
            ema_p.copy_(ema_p * self.decay + p.data * (1 - self.decay))


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------
def save_checkpoint(output_dir, epoch, G, D, opt_G, opt_D, step):
    path = Path(output_dir) / f"ckpt_epoch{epoch:03d}.pt"
    # Unwrap DDP / torch.compile wrappers
    g_state = getattr(G, "_orig_mod", G)
    g_state = getattr(g_state, "module", g_state)
    d_state = getattr(D, "_orig_mod", D)
    d_state = getattr(d_state, "module", d_state)

    torch.save({
        "epoch"    : epoch,
        "step"     : step,
        "G"        : g_state.state_dict(),
        "D"        : d_state.state_dict(),
        "opt_G"    : opt_G.state_dict(),
        "opt_D"    : opt_D.state_dict(),
    }, path)
    logging.getLogger(__name__).info(f"Saved checkpoint: {path}")


def load_checkpoint(path, G, D, opt_G, opt_D, rank=0):
    ckpt = torch.load(path, map_location="cpu")
    g_state = getattr(G, "_orig_mod", G)
    g_state = getattr(g_state, "module", g_state)
    d_state = getattr(D, "_orig_mod", D)
    d_state = getattr(d_state, "module", d_state)

    g_state.load_state_dict(ckpt["G"])
    d_state.load_state_dict(ckpt["D"])
    opt_G.load_state_dict(ckpt["opt_G"])
    opt_D.load_state_dict(ckpt["opt_D"])
    if rank == 0:
        logging.getLogger(__name__).info(f"Resumed from {path} (epoch {ckpt['epoch']})")
    return ckpt["epoch"] + 1


# ---------------------------------------------------------------------------
# Image Logging
# ---------------------------------------------------------------------------
def log_images(fake, real, output_dir, step, n=4):
    """Save side-by-side real/fake grid as PNG."""
    def to_uint8(t):
        return ((t.clamp(-1, 1) + 1) / 2 * 255).byte().cpu()

    imgs = []
    for i in range(min(n, fake.shape[0])):
        imgs.extend([to_uint8(real[i:i+1]), to_uint8(fake[i:i+1])])

    grid = vutils.make_grid(
        torch.cat(imgs), nrow=n*2, padding=2, normalize=False
    ).float() / 255.0

    save_path = Path(output_dir) / f"samples_step{step:07d}.png"
    vutils.save_image(grid, save_path)
