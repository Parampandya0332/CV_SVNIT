# Restormer Fine-Tuning for NTIRE 2026 Image Denoising (σ = 50)

This repository contains the **training and evaluation pipeline** used for the **NTIRE 2026 Image Denoising Challenge (Gaussian noise σ = 50)**.

The approach is based on the **Restormer architecture**, a transformer-based model designed for high-resolution image restoration tasks.

The model is **fine-tuned on the DIV2K dataset** and evaluated using tiled inference to handle high-resolution images efficiently.

---

# 1. Repository Structure

```
CV_SVNIT
│
├── training.py
├── evaluation.py
├── README.md
├── requirements.txt
│
├── Restormer/              # official Restormer implementation
│
├── test_images/            # input noisy images
│
└── results/                # output denoised images
```

---

# 2. Prerequisites

Recommended environment:

Python ≥ 3.8
PyTorch ≥ 2.0
CUDA-enabled GPU (recommended)

---

# 3. Install Dependencies

Install required libraries:

```
pip install -r requirements.txt
```

or manually:

```
pip install torch torchvision
pip install einops timm lmdb pillow numpy
```

---

# 4. Clone Repository

```
git clone https://github.com/Parampandya0332/CV_SVNIT.git
cd CV_SVNIT
```

Clone the official Restormer repository:

```
git clone https://github.com/swz30/Restormer.git
```

---

# 5. Dataset Preparation (Training)

Training uses the **DIV2K dataset**.

Download from:

https://data.vision.ee.ethz.ch/cvl/DIV2K/

Expected structure:

```
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
```

---

# 6. Training Configuration

Noise level: σ = 50

Patch size: 256 × 256

Training iterations: 20000

Batch size: 1

Optimizer: AdamW

Learning rate: 1e-4

Weight decay: 1e-4

Learning rate scheduler: Cosine Annealing

Loss function: Charbonnier Loss

Mixed precision training: Enabled (AMP)

---

# 7. Training

Run training:

```
python training.py
```

Training pipeline:

1. Load DIV2K images
2. Random crop 256×256 patches
3. Add Gaussian noise (σ=50)
4. Train Restormer
5. Validate every 500 iterations
6. Save best model based on PSNR

---

# 8. Evaluation / Inference

Place noisy test images in:

```
test_images/
```

Run inference:

```
python evaluation.py
```

Outputs will be saved in:

```
results/
```

---

# 9. Model Checkpoint Loading

The evaluation script supports two checkpoint formats.

### Pure model weights

```
model.load_state_dict(weights)
```

### Training checkpoints

```
{
 "model": state_dict,
 "optimizer": ...
 "scheduler": ...
}
```

The evaluation script automatically detects the checkpoint format and loads the correct weights.

---

# 10. Evaluation Strategy

High-resolution images are processed using **tiled inference**.

Tile size: 512

Tile overlap: 32

Steps:

1. Divide image into overlapping tiles
2. Pad tiles so dimensions are divisible by 8
3. Run Restormer on each tile
4. Merge overlapping outputs

This avoids GPU memory overflow and prevents boundary artifacts.

---

# 11. Runtime Measurement

Runtime is measured automatically during inference.

The script prints:

```
Runtime per image: X seconds
```

This metric is required for NTIRE evaluation.

---

# 12. Testing the Repository

To verify the repository runs correctly:

```
git clone https://github.com/Parampandya0332/CV_SVNIT.git
cd CV_SVNIT

pip install -r requirements.txt

python evaluation.py
```

If the script finishes successfully and outputs images inside `results/`, the repository is correctly configured.

---

# 13. Acknowledgement

This project is based on the official Restormer implementation.

Restormer: Efficient Transformer for High-Resolution Image Restoration

Official repository:

https://github.com/swz30/Restormer
