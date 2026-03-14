# Lightweight U-Net for Face Parsing (CelebAMask-HQ)

This repository contains a compact deep learning implementation of a **U-Net architecture for facial semantic segmentation**.

The objective of the project was to design and train an **efficient face parsing model** capable of segmenting facial regions while operating under strict computational constraints.

---

# Task Description

Face parsing is a semantic segmentation problem where each pixel in a facial image is assigned to a semantic class such as:

* skin
* hair
* eyes
* eyebrows
* lips
* nose
* accessories

The project used a subset of the **CelebAMask-HQ dataset**, which provides detailed segmentation masks for facial components.

The goal was to train a model that can correctly predict these classes for every pixel in an image.

---

# Project Constraints

Unlike many modern segmentation approaches, this project had several **explicit restrictions** that strongly influenced the model design:

### Parameter Limit

The model was required to contain **fewer than 1.82 million trainable parameters**.

### Training Data

Only the **provided dataset subset** was allowed:

* 1000 training images
* 100 validation images

No external datasets could be used.

### Training Requirements

The model had to be trained:

* **from scratch**
* without pretrained backbones
* without transfer learning
* without knowledge distillation
* without test-time augmentation

### Hardware Environment

All experiments were conducted on:

* **Google Colab**
* **NVIDIA T4 GPU**

These constraints required a model that balances **accuracy, efficiency, and reproducibility**.

---

# Model Architecture

The model is based on a **compact U-Net encoder-decoder architecture**.

Architecture overview:

Encoder

* two convolutional blocks
* each block: two 3×3 convolutions + ReLU
* 2×2 max pooling

Bottleneck

* deeper feature representation

Decoder

* transposed convolutions for upsampling
* skip connections with encoder layers

Output

* 1×1 convolution producing **19 segmentation classes**

Total parameters:

**467,123**

This corresponds to only **~26% of the allowed parameter budget**, ensuring compliance with the project requirements.

---

# Data Preprocessing

The CelebAMask-HQ masks are stored as **RGB color images**, where each color represents a facial class.

However, segmentation models require **integer class labels**.

Therefore the preprocessing pipeline performs:

1. Extraction of unique RGB colors
2. Mapping of each color to a class ID
3. Conversion of masks to integer arrays
4. Caching masks as `.npy` files

This conversion prevents invalid class indices and significantly speeds up data loading during training.

---

# Training Procedure

Images are resized to:

256 × 256

This resolution provides a good trade-off between:

* computational cost
* segmentation quality

The training configuration:

Optimizer
AdamW

Learning Rate
1e-3

Scheduler
Cosine Annealing

Epochs
50

Batch Size
8

Data Augmentation

* random horizontal flip
* light color jitter

---

# Loss Function

A **combined loss function** was used:

CrossEntropy Loss
for pixel-level classification

Dice Loss
for improving overlap between predicted and true regions

Dice loss helps particularly with **small classes** such as:

* eyes
* lips
* eyebrows

which are otherwise underrepresented in the dataset.

---

# Training Observations

Several implementation challenges were encountered during development:

### Mask Encoding Errors

Initial training attempts produced **CUDA device-side assertion errors** due to incorrect class indices caused by RGB mask encoding.

This was solved by explicitly mapping RGB colors to class IDs.

### Training Resolution

Training at the original **512×512 resolution** proved unstable and slow.

Reducing the resolution to **256×256** reduced training time per epoch from ~80 seconds to ~25 seconds while maintaining comparable accuracy.

### GPU Memory Constraints

Memory allocation errors occurred during early experiments.

The solution was to:

* reduce batch size from 16 to 8
* enable `pin_memory` in the DataLoader

These changes stabilized training.

---

# Results

Final validation performance:

Pixel Accuracy
≈ **84%**

Mean Dice (F1) Score
≈ **0.62**

The model performs well on large regions such as:

* skin
* hair
* background

Smaller objects such as:

* glasses
* earrings

remain more difficult due to limited training samples.

Despite its compact size, the model produces stable segmentation results.

---

# Repository Structure

DL_final.ipynb
Full training and evaluation pipeline

model_best.pth
Best performing model checkpoint

N2505076G_SimonEnkel_project1.docx
Detailed project report

requirements.txt
Required Python dependencies

---

# How to Run

Install dependencies:

pip install -r requirements.txt

Then open the notebook:

DL_final.ipynb

Run the cells sequentially to reproduce training and evaluation.

---

# Key Takeaways

This project demonstrates that:

* efficient architectures can achieve competitive results
* careful preprocessing and debugging are critical for segmentation tasks
* smaller models can perform well when training pipelines are well designed

The final lightweight U-Net achieves solid segmentation accuracy while remaining computationally efficient and reproducible.

---

# Author

Simon Enkel


Course
CE7454 – Deep Learning for Data Science
