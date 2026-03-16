# Restormer Fine-Tuning for NTIRE 2026 Image Denoising (σ=50)

This repository contains the training and evaluation code for our submission to the NTIRE 2026 Image Denoising Challenge (Gaussian noise σ=50).

The method is based on the Restormer architecture with full fine-tuning on the DIV2K dataset.

---

Architecture

We use the official Restormer model configuration:

Input channels: 3  
Output channels: 3  
Embedding dimension: 48  

Transformer blocks:

Encoder:
4 → 6 → 6 → 8 blocks

Heads:
1 → 2 → 4 → 8

FFN expansion factor:
2.66

LayerNorm:
BiasFree

The model is initialized using pretrained weights trained for Gaussian color denoising σ=50.

---

Training Strategy

Training dataset:
DIV2K Train HR

Validation dataset:
DIV2K Valid HR

Patch size:
256 × 256

Noise model:
Gaussian noise σ = 50

Training iterations:
20000

Batch size:
1

Loss function:
Charbonnier Loss

Optimizer:
AdamW

Learning rate:
1e-4

Weight decay:
1e-4

Learning rate scheduler:
Cosine Annealing

Mixed precision training:
Yes (AMP)

Validation is performed every 500 iterations using 100 fixed patches.

---

Modifications to Restormer

The Restormer architecture itself was not modified.

The following training strategies were applied:

1. Fine-tuning using pretrained σ=50 weights  
2. Random patch training (256×256)  
3. Gaussian noise generation during training  
4. Charbonnier loss instead of L2  
5. Cosine annealing learning rate schedule  
6. Mixed precision training for efficiency

---

Evaluation Strategy

Evaluation is performed using tiled inference to process high-resolution images.

Tile size:
512

Tile overlap:
32

Reflection padding is applied to ensure that tiles are divisible by 8, which is required by the Restormer architecture.

Outputs from overlapping tiles are averaged to avoid boundary artifacts.

---

Runtime

Average runtime per image is computed during evaluation.

Inference is performed using GPU acceleration.

---

Repository Structure

training.py  
evaluation.py  
Restormer/ (official repository)

---

Usage

Training
