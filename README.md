# TryOnGAN: Body-Aware Try-On via StyleGAN

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-TryOnGAN-blue.svg)](https://arxiv.org/abs/2104.03222)

A professional, high-performance PyTorch implementation of the **TryOnGAN** architecture for virtual garment try-on. This repository implements a pose-conditioned and segmentation-aware StyleGAN2 backbone to synthesize high-fidelity images of a person wearing a target garment, preserving both user identity and fine garment textures.

## Table of Contents
1. [Architecture Overview](#architecture-overview)
2. [Repository Structure](#repository-structure)
3. [Installation & Setup](#installation--setup)
4. [Dataset Preparation](#dataset-preparation)
5. [Training Pipeline](#training-pipeline)
6. [Inference & Latent Optimization](#inference--latent-optimization)
7. [Research & Engineering Extensions](#research--engineering-extensions)

---

## Architecture Overview

This implementation faithfully reproduces the core contributions of TryOnGAN:

* **Generative Backbone:** A StyleGAN2-based generator utilizing weight demodulation (AdaIN), continuous skip connections, and noise injection for stochastic detail generation.
* **Disentangled Encoders:**
    * `PoseEncoder`: Processes sparse 18-channel OpenPose Gaussian heatmaps into the $W$ latent space.
    * `ClothEncoder`: Employs a custom Lite Feature Pyramid Network (FPN) neck to aggregate multi-scale features (P3/P4/P5). This prevents the loss of high-frequency garment details (stripes, logos) prior to global pooling.
* **Segmentation-Aware Discriminator:** The discriminator ingests a concatenation of the RGB image and a binary body segmentation mask. This forces the model to learn topological correctness (e.g., shirts go on the torso), preventing standard GAN mode collapse into unstructured textures.
* **Latent Space Injection:** Both encoded styles are concatenated, passed through an 8-layer Mapping Network, and broadcasted to the $W+$ space (18 × 512 dimensions) to modulate the synthesis blocks.

---

## Repository Structure

| Component | File | Description |
| :--- | :--- | :--- |
| **Data Loader** | `dataset.py` | Handles VITON-HD ingestion, OpenPose heatmap rendering (σ=6px), and LIP mask binarization. |
| **Generator** | `models/generator.py` | StyleGAN2 synthesis network, Mapping Network, and Modulated Convolutions. |
| **Encoders** | `models/encoder.py` | ResNet-style backbones. Includes the `LiteFPN` for texture preservation. |
| **Discriminator**| `models/discriminator.py` | Segmentation-conditioned multi-scale network with MiniBatch StdDev. |
| **Loss Functions**| `models/losses.py` | Non-saturating logistic loss, Lazy R1 Penalty, and VGG16 Perceptual loss. |
| **Training** | `train.py` | DDP-enabled training script utilizing AMP (`GradScaler`) and `torch.compile`. |
| **Inference** | `inference.py` | Synthesis script featuring GAN-inversion style latent optimization. |
| **Utilities** | `utils.py` | Exponential Moving Average (EMA) tracking, checkpointing, and logging. |

---

## Installation & Setup

Ensure you have a modern GPU environment (CUDA 11.8+ recommended) and Python 3.10+.

```bash
# Clone the repository
git clone <repository-url>
cd tryongan-pytorch

# Install core dependencies
pip install torch torchvision numpy Pillow
```

---

## Dataset Preparation

The pipeline is configured for the **VITON-HD** dataset. Your data directory must match the following structure precisely. Images should be pre-processed to 512x512 or 1024x768.

```text
{data_root}/
  train/
    image/          # Full-body person photos (.jpg/.png)
    cloth/          # Target garment images (.jpg/.png)
    openpose-json/  # OpenPose 18-keypoint data (.json)
    image-parse/    # LIP segmentation maps (.png)
  test/
    ...             # Same structure for evaluation
```

---

## Training Pipeline

The training loop is highly optimized for modern hardware, featuring **Automatic Mixed Precision (AMP)**, **`torch.compile`** for kernel fusion, and **Lazy Regularization** (applying the R1 penalty every 16 steps to reduce backward pass overhead by ~50%).

### Single GPU Training
```bash
python train.py \
    --data_root ./data/viton-hd \
    --output_dir ./checkpoints/tryon_experiment \
    --img_size 512 \
    --batch_size 4 \
    --use_compile \
    --epochs 100
```

### Multi-GPU Distributed Data Parallel (DDP)
```bash
torchrun --nproc_per_node=2 train.py \
    --data_root ./data/viton-hd \
    --output_dir ./checkpoints/tryon_experiment \
    --img_size 512 \
    --batch_size 4 \
    --grad_accum 4
```

---

## Inference & Latent Optimization

The inference script supports two modes: **Direct Feedforward** (fast) and **Latent Optimization** (high fidelity). 

Latent optimization performs gradient descent on the $W$ vectors to minimize the perceptual distance between the generated output and the target person, ensuring strict identity preservation (similar to GAN Inversion).

```bash
# High-Fidelity Inference (with optimization)
python inference.py \
    --ckpt ./checkpoints/tryon_experiment/ckpt_epoch099.pt \
    --person ./data/viton-hd/test/image/person_01.jpg \
    --cloth ./data/viton-hd/test/cloth/garment_01.jpg \
    --pose ./data/viton-hd/test/openpose-json/person_01_keypoints.json \
    --out ./outputs/result_01.jpg \
    --optimize
```

---

## Research & Engineering Extensions

For engineers and researchers looking to productionize or extend this codebase, consider the following architectural improvements:

### 1. Production-Grade System Architecture
To deploy this as a scalable microservice, wrap the PyTorch inference engine within an asynchronous worker pool. You can utilize **Node.js** with **TypeScript** to build the core API. Use **Redis (Upstash)** and **BullMQ** to queue image generation tasks for Dockerized GPU workers. State and user catalogs can be managed cleanly using **Prisma** paired with **PostgreSQL**.

### 2. Low-Level Inference Optimization
The 200-step latent optimization loop in `inference.py` is computationally expensive. Profiling the autograd graph and porting the critical mathematical operations of the synthesis blocks to **C++ / CUDA** could dramatically reduce latency, pushing the system closer to real-time interactive try-on.

### 3. Latent Space Disentanglement Analysis
To quantitatively evaluate how well the model separates pose from clothing, you can apply dimensionality reduction algorithms like **Principal Component Analysis (PCA)** or t-SNE to the extracted $W+$ vectors. Evaluating the clustering of these vectors can help refine the encoders. 

### 4. Advanced Evaluation Metrics
Standard GAN metrics (FID) don't fully capture "try-on accuracy". You can train an auxiliary classifier to detect "garment match" and evaluate the generator using classification metrics. By plotting the **ROC-AUC curves** of this classifier against different generator checkpoint outputs, you can rigorously quantify texture preservation.

### 5. Explicit Spatial Warping
While the FPN captures texture, complex graphics often warp incorrectly. Introducing a spatial transformer network (like a Thin-Plate Spline warper) before the `ClothEncoder` can align the garment to the target body segmentation mask explicitly.
