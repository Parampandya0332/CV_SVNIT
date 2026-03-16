# Restormer Fine-Tuning for NTIRE 2026 Image Denoising (σ = 50)

This repository contains the **training and evaluation code** for our submission to the **NTIRE 2026 Image Denoising Challenge (Gaussian Noise σ = 50)**.

Our method is based on the **Restormer (Efficient Transformer for High-Resolution Image Restoration)** architecture and uses **full fine-tuning on the DIV2K dataset**.

The repository provides a clean and reproducible implementation including:

* Training pipeline
* Validation strategy
* Tiled inference for high-resolution images
* Runtime evaluation

---

# 1. Prerequisites

Before running the code, ensure the following software is installed.

## System Requirements

* Python ≥ 3.8
* PyTorch ≥ 2.0
* CUDA-enabled GPU (recommended for training and inference)
* Linux / Windows / Kaggle environment

---

# 2. Python Dependencies

Install the required libraries:

pip install torch torchvision
pip install einops timm lmdb
pip install pillow numpy

These libraries are required for:

* Transformer operations
* Image processing
* Dataset loading
* Training and inference

---

# 3. Repository Setup

Clone this repository:

git clone https://github.com/<your-username>/<your-repository>.git
cd <your-repository>

Clone the official Restormer implementation inside the repository:

git clone https://github.com/swz30/Restormer.git

The Restormer folder provides the architecture implementation used by this project.

---

# 4. Dataset Preparation

Training uses the **DIV2K dataset**.

Download the dataset from:

https://data.vision.ee.ethz.ch/cvl/DIV2K/

Expected dataset structure:

datasets/

├── DIV2K_train_HR
│     ├── 0001.png
│     ├── 0002.png
│     └── ...
│
└── DIV2K_valid_HR
├── 0801.png
├── 0802.png
└── ...

---

# 5. Model Architecture

We use the **Restormer architecture** with the following configuration:

Input Channels: 3
Output Channels: 3
Embedding Dimension: 48

Encoder Blocks:

[4, 6, 6, 8]

Attention Heads:

[1, 2, 4, 8]

FFN Expansion Factor:

2.66

LayerNorm Type:

BiasFree

The model is initialized using **pretrained weights trained for Gaussian denoising with σ = 50**.

---

# 6. Training Strategy

Training configuration:

Noise Level: σ = 50
Patch Size: 256 × 256
Iterations: 20000
Batch Size: 1

Optimizer: AdamW

Learning Rate: 1e-4

Weight Decay: 1e-4

Learning Rate Scheduler: Cosine Annealing

Loss Function: Charbonnier Loss

Mixed Precision Training: Enabled (AMP)

---

# 7. Data Preparation During Training

During training:

1. Images are randomly cropped into **256 × 256 patches**
2. Gaussian noise with **σ = 50** is generated dynamically
3. Noisy patches are used as input to the network
4. Clean patches are used as supervision targets

This approach increases the effective dataset size and improves model generalization.

---

# 8. Validation Strategy

Validation is performed using:

100 fixed patches extracted from the DIV2K validation set.

Validation occurs every **500 training iterations**.

PSNR is computed to monitor denoising performance.

The model achieving the **highest validation PSNR** is saved as the best checkpoint.

---

# 9. Modifications Applied to Restormer

The **core Restormer architecture was not modified**.

However, the following training strategies were applied:

1. Full fine-tuning starting from pretrained σ=50 weights
2. Random patch training (256×256 patches)
3. On-the-fly Gaussian noise generation
4. Charbonnier loss for stable optimization
5. Cosine annealing learning rate scheduling
6. Mixed precision training for faster GPU computation

These modifications improve training stability and denoising performance.

---

# 10. Training Instructions

Run training using:

python training.py

Training steps:

1. Load DIV2K training dataset
2. Generate noisy image patches (σ = 50)
3. Train Restormer using mixed precision
4. Validate every 500 iterations
5. Save the best performing model

Saved checkpoints include:

best_model.pth
final_model.pth

---

# 11. Evaluation / Inference

Place test images inside the folder:

test_images/

Run inference using:

python evaluation.py

The script will:

1. Load the trained Restormer model
2. Process test images using tiled inference
3. Save denoised outputs

Outputs will be saved in:

results/

---

# 12. Evaluation Strategy

High-resolution images are processed using **tiled inference** to avoid GPU memory overflow.

Tile configuration:

Tile Size: 512
Tile Overlap: 32

Steps used during inference:

1. Each image is divided into overlapping tiles
2. Tiles are padded to be divisible by 8 (required by Restormer)
3. Each tile is denoised independently
4. Overlapping outputs are averaged to remove boundary artifacts

This strategy enables efficient processing of large images.

---

# 13. Runtime Measurement

The evaluation script measures:

Average runtime per image.

Inference is performed using **GPU acceleration**.

Runtime information is used for NTIRE submission requirements.

---

# 14. Repository Structure

repository/

├── Restormer/
│
├── training.py
├── evaluation.py
├── README.md
│
├── datasets/
│     ├── DIV2K_train_HR
│     └── DIV2K_valid_HR
│
├── test_images/
│
└── results/

---

# 15. Acknowledgement

This work is based on the official Restormer implementation.

Restormer: Efficient Transformer for High-Resolution Image Restoration

Official repository:

https://github.com/swz30/Restormer
