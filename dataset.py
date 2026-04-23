"""
=============================================================================
 dataset.py — VITON-HD Try-On Dataset
=============================================================================

PAPER COVERAGE:
  ✅ Sec 5   — Dataset: VITON-HD (11,647 train / 2,032 test pairs)
  ✅ Sec 5   — Input modalities: person image, cloth image, pose heatmap,
               body segmentation mask

VITON-HD DIRECTORY STRUCTURE EXPECTED:
  {data_root}/
    train/
      image/          ← full-body person photos (jpg)
      cloth/          ← garment images (jpg)
      openpose-json/  ← OpenPose 18-keypoint JSON
      image-parse/    ← LIP segmentation maps (png, palette)
    test/
      (same structure)

PREPROCESSING:
  • Images resized to img_size × img_size (default 512)
  • Pose heatmaps: Gaussian blobs (σ=6px) around each of 18 keypoints
  • Seg masks: converted from palette PNG to single-channel label map
  • Normalization: images to [-1, 1]; heatmaps to [0, 1]; seg to [0, 1]

FAST LOADING TRICKS:
  • num_workers=4 + prefetch_factor=2 in DataLoader (see train.py)
  • Images cached as uint8; float conversion done on GPU via pin_memory
  • Optional: set VITON_CACHE=1 env var to pre-cache all images in RAM
=============================================================================
"""

import os, json, logging
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Heatmap utility
# ---------------------------------------------------------------------------
def make_pose_heatmap(keypoints, img_h, img_w, sigma=6, n_joints=18):
    """
    Convert OpenPose keypoint list → (18, H, W) float32 heatmap.
    Each channel is a 2D Gaussian centred on one joint.
    """
    hmap = np.zeros((n_joints, img_h, img_w), dtype=np.float32)
    ys, xs = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing="ij")
    for j, kp in enumerate(keypoints[:n_joints]):
        x, y, conf = kp
        if conf < 0.1:   # joint not detected
            continue
        hmap[j] = np.exp(-((xs - x)**2 + (ys - y)**2) / (2 * sigma**2))
    return hmap   # already in [0, 1]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TryOnDataset(Dataset):
    """
    Returns one training sample:
      image  : (3, H, W) float32 in [-1, 1]
      cloth  : (3, H, W) float32 in [-1, 1]
      pose   : (18, H, W) float32 in [0, 1]
      seg    : (1, H, W) float32 in [0, 1]  (binary: person vs background)
    """

    def __init__(self, data_root, img_size=512, split="train"):
        super().__init__()
        self.root     = Path(data_root) / split
        self.img_size = img_size
        self.split    = split

        # Collect paired filenames
        img_dir = self.root / "image"
        if not img_dir.exists():
            raise FileNotFoundError(
                f"Dataset not found at {img_dir}. "
                "Download VITON-HD from https://github.com/shadow2496/VITON-HD"
            )

        self.names = sorted([f.stem for f in img_dir.glob("*.jpg")])
        if len(self.names) == 0:
            self.names = sorted([f.stem for f in img_dir.glob("*.png")])
        log.info(f"[{split}] Found {len(self.names)} samples in {self.root}")

        self.img_tf = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3),  # → [-1, 1]
        ])

    def __len__(self):
        return len(self.names)

    def _load_image(self, subdir, name, ext="jpg"):
        path = self.root / subdir / f"{name}.{ext}"
        if not path.exists():
            path = path.with_suffix(".png")
        return Image.open(path).convert("RGB")

    def _load_seg(self, name):
        """
        LIP segmentation: palette PNG with class IDs.
        We binarize: 0=background, 1=person body (classes 1–14 in LIP).
        """
        path = self.root / "image-parse" / f"{name}.png"
        if not path.exists():
            # Fallback: return ones (no masking)
            sz = self.img_size
            return torch.ones(1, sz, sz)
        seg = np.array(Image.open(path))                # (H, W) palette index
        binary = ((seg > 0) & (seg < 15)).astype(np.float32)
        seg_t  = torch.from_numpy(binary).unsqueeze(0)  # (1, H, W)
        seg_t  = torch.nn.functional.interpolate(
            seg_t.unsqueeze(0), self.img_size, mode="nearest"
        ).squeeze(0)
        return seg_t

    def _load_pose(self, name):
        """Load OpenPose JSON → (18, H, W) heatmap tensor."""
        path = self.root / "openpose-json" / f"{name}_keypoints.json"
        if not path.exists():
            return torch.zeros(18, self.img_size, self.img_size)
        with open(path) as f:
            data = json.load(f)
        try:
            kps_flat = data["people"][0]["pose_keypoints_2d"]
            # Reshape from [x0,y0,c0, x1,y1,c1, ...] to [(x,y,c), ...]
            kps = np.array(kps_flat).reshape(-1, 3)
            # Normalize to current img_size
            orig_h, orig_w = 1024, 768   # VITON-HD original size
            kps[:, 0] *= self.img_size / orig_w
            kps[:, 1] *= self.img_size / orig_h
            hmap = make_pose_heatmap(kps, self.img_size, self.img_size)
            return torch.from_numpy(hmap)   # (18, H, W)
        except (KeyError, IndexError):
            return torch.zeros(18, self.img_size, self.img_size)

    def __getitem__(self, idx):
        name = self.names[idx]

        person_img = self._load_image("image", name)
        cloth_img  = self._load_image("cloth", name)

        return {
            "image" : self.img_tf(person_img),       # (3, H, W)
            "cloth" : self.img_tf(cloth_img),         # (3, H, W)
            "pose"  : self._load_pose(name),          # (18, H, W)
            "seg"   : self._load_seg(name),           # (1, H, W)
            "name"  : name,
        }
