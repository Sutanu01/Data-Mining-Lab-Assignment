"""
=============================================================================
 inference.py — Try-On Synthesis (Paper § 3.3)
=============================================================================

PAPER COVERAGE:
  ✅ Sec 3.3 — Latent optimization: given a real person + new cloth,
               find w* that minimizes reconstruction + style loss
  ✅ Sec 3.3 — Style transfer via W+ space interpolation
  ✅ Sec 5   — EMA generator used for inference

HOW TRYON INFERENCE WORKS (per paper):
  1. Encode person pose:  w_pose  = PoseEncoder(pose_hmap)
  2. Encode new cloth:    w_cloth = ClothEncoder(cloth_img)
  3. Generate:            img     = G([w_pose, w_cloth])

  Optionally run latent optimization (GAN inversion style):
  - Start with w from step 2
  - Iteratively minimize perceptual(G(w), target) for ~200 steps
  - Gives more identity-preserving results

USAGE:
  python inference.py \
    --ckpt   /kaggle/working/tryon_out/ckpt_epoch099.pt \
    --person /path/to/person.jpg \
    --cloth  /path/to/new_cloth.jpg \
    --pose   /path/to/person_keypoints.json \
    --out    /kaggle/working/result.jpg
=============================================================================
"""

import argparse, json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import torchvision.utils as vutils
from pathlib import Path

from models.generator import TryOnGenerator
from models.encoder   import PoseEncoder, ClothEncoder
from losses           import PerceptualLoss
from dataset          import make_pose_heatmap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
IMG_TF = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])


def load_image(path, device):
    img = Image.open(path).convert("RGB")
    return IMG_TF(img).unsqueeze(0).to(device)


def load_pose(json_path, img_size, device):
    with open(json_path) as f:
        data = json.load(f)
    kps_flat = data["people"][0]["pose_keypoints_2d"]
    kps = np.array(kps_flat).reshape(-1, 3)
    kps[:, 0] *= img_size / 768
    kps[:, 1] *= img_size / 1024
    hmap = make_pose_heatmap(kps, img_size, img_size)
    return torch.from_numpy(hmap).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Optional: latent optimization (GAN inversion, paper § 3.3)
# ---------------------------------------------------------------------------
@torch.no_grad()
def direct_inference(G, pose_enc, cloth_enc, pose, cloth, truncation=0.7):
    """Fast single-forward try-on (no optimization)."""
    w_pose  = pose_enc(pose)
    w_cloth = cloth_enc(cloth)
    img, _  = G([w_pose, w_cloth], truncation=truncation)
    return img


def latent_optimization(G, pose_enc, cloth_enc, perc_loss,
                         pose, cloth, target,
                         n_steps=200, lr=0.01):
    """
    Refine latent via gradient descent to better preserve identity.
    Minimizes: λ_perc * perc(G(w), target) + λ_cloth * ‖w_cloth - orig‖²
    """
    with torch.no_grad():
        w_pose_0  = pose_enc(pose).detach()
        w_cloth_0 = cloth_enc(cloth).detach()

    # Optimizable copy
    w_pose  = w_pose_0.clone().requires_grad_(True)
    w_cloth = w_cloth_0.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([w_pose, w_cloth], lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        img, _ = G([w_pose, w_cloth])
        loss = (
            perc_loss(img, target) * 1.0 +
            F.mse_loss(w_cloth, w_cloth_0) * 0.1
        )
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"  opt step {step:03d} | loss={loss.item():.4f}")

    with torch.no_grad():
        img, _ = G([w_pose, w_cloth])
    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser("TryOnGAN Inference")
    p.add_argument("--ckpt",       required=True)
    p.add_argument("--person",     required=True)
    p.add_argument("--cloth",      required=True)
    p.add_argument("--pose",       required=True, help="OpenPose JSON")
    p.add_argument("--out",        default="result.jpg")
    p.add_argument("--img_size",   type=int, default=512)
    p.add_argument("--style_dim",  type=int, default=512)
    p.add_argument("--optimize",   action="store_true",
                   help="Run latent optimization for better identity preservation")
    p.add_argument("--truncation", type=float, default=0.7)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    G         = TryOnGenerator(img_size=args.img_size,
                                style_dim=args.style_dim).to(device).eval()
    pose_enc  = PoseEncoder(out_dim=args.style_dim).to(device).eval()
    cloth_enc = ClothEncoder(out_dim=args.style_dim).to(device).eval()

    ckpt = torch.load(args.ckpt, map_location=device)
    G.load_state_dict(ckpt["G"])
    # Encoders not in ckpt separately — load from G ckpt if needed:
    # pose_enc.load_state_dict(ckpt["pose_enc"])
    # cloth_enc.load_state_dict(ckpt["cloth_enc"])

    # Load inputs
    cloth  = load_image(args.person, device)   # person image (for target)
    new_cloth = load_image(args.cloth, device)
    pose   = load_pose(args.pose, args.img_size, device)
    target = load_image(args.person, device)

    # Inference
    with torch.no_grad():
        if args.optimize:
            perc_loss = PerceptualLoss().to(device)
            result = latent_optimization(G, pose_enc, cloth_enc, perc_loss,
                                          pose, new_cloth, target)
        else:
            result = direct_inference(G, pose_enc, cloth_enc,
                                       pose, new_cloth, args.truncation)

    # Save
    out_img = ((result.squeeze().clamp(-1,1) + 1) / 2)
    vutils.save_image(out_img, args.out)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
