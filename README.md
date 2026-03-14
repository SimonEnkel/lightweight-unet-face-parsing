# Lightweight U-Net for Face Parsing (CelebAMask-HQ)


This project implements a compact **U-Net architecture** for semantic segmentation of facial regions using a subset of the **CelebAMask-HQ dataset**.

The goal was to build an efficient model capable of segmenting **19 semantic facial regions** while staying under strict computational constraints.

---

# Overview

Face parsing is a semantic segmentation task where each pixel of a facial image is assigned a class label such as skin, hair, eyes, or lips.

Applications include:

* facial editing
* augmented reality
* emotion recognition
* avatar generation

This repository contains a complete deep learning pipeline:

* dataset preprocessing
* U-Net model architecture
* training and evaluation
* metric tracking
* segmentation prediction

---

# Model Architecture

The model follows a **lightweight encoder–decoder U-Net architecture**:

Encoder
• two convolution blocks
• ReLU activation
• max pooling

Bottleneck
• feature compression and representation learning

Decoder
• transposed convolutions for upsampling
• skip connections to preserve spatial information

Final layer
• 1×1 convolution mapping features to **19 segmentation classes**

Total parameters: **467,123**

---

# Training Setup

Dataset:
CelebAMask-HQ (subset)

Image resolution:
256 × 256 during training

Optimizer:
AdamW

Learning rate scheduler:
Cosine annealing

Loss function:
Combined loss

* CrossEntropyLoss
* Dice Loss

Training epochs:
50

Hardware:
Google Colab (NVIDIA T4 GPU)

---

# Results

Validation performance:

Pixel Accuracy: ~84%
Dice (F1) Score: ~0.62

The model achieved stable convergence while maintaining a very small parameter count compared to standard segmentation architectures.

Despite the compact design, the network successfully captures large facial structures such as skin, hair, and mouth regions.

---

# Repository Structure

DL_final.ipynb
Main notebook containing the full training and evaluation pipeline

model_best.pth
Best performing trained model checkpoint

report.pdf
Full project report and technical documentation

requirements.txt
Python dependencies required to run the project

---

# How to Run

Install dependencies:

pip install -r requirements.txt

Open the notebook:

DL_final.ipynb

Run all cells to reproduce training and evaluation.

---

# Future Improvements

Possible extensions include:

• pretrained encoder backbones
• lightweight attention modules
• higher resolution training
• improved class balancing strategies

---

# Author

Simon Enkel

---

# License

This project is intended for educational and research purposes.
