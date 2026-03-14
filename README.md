# Lightweight U-Net for Face Parsing

A compact semantic segmentation model that parses facial regions at the pixel level — built entirely from scratch, no pretrained weights, no external data.

**467K parameters · 84% pixel accuracy · 0.62 mean Dice score**

---

## What it does

Given a face image, the model assigns each pixel to one of 19 semantic classes:

`skin` · `hair` · `left eye` · `right eye` · `left eyebrow` · `right eyebrow` · `nose` · `upper lip` · `lower lip` · `mouth` · `left ear` · `right ear` · `glasses` · `earrings` · `necklace` · `neck` · `cloth` · `background` · `hat`

<!-- Add prediction examples here: input image → segmentation mask -->

---

## Design Constraints

This project was built under strict constraints that shaped every architectural and training decision:

| Constraint | Value |
|---|---|
| Max parameters | 1,820,000 |
| Training images | 1,000 |
| Validation images | 100 |
| Pretrained weights | ❌ not allowed |
| Transfer learning | ❌ not allowed |
| External datasets | ❌ not allowed |
| Hardware | Google Colab · NVIDIA T4 |

The final model uses only **26% of the allowed parameter budget** (467,123 params), leaving significant headroom while still achieving competitive results.

---

## Architecture

A custom compact U-Net encoder-decoder, designed to balance accuracy and efficiency within the parameter budget.

```
Input (3 × 256 × 256)
    │
    ├─ Encoder Block 1: Conv 3×3 → ReLU → Conv 3×3 → ReLU   [32 ch]
    ├─ MaxPool 2×2
    ├─ Encoder Block 2: Conv 3×3 → ReLU → Conv 3×3 → ReLU   [64 ch]
    ├─ MaxPool 2×2
    │
    ├─ Bottleneck: Conv 3×3 → ReLU → Conv 3×3 → ReLU        [128 ch]
    │
    ├─ ConvTranspose 2×2 + skip connection from Enc2          [64 ch]
    ├─ Decoder Block 1: Conv 3×3 → ReLU → Conv 3×3 → ReLU
    ├─ ConvTranspose 2×2 + skip connection from Enc1          [32 ch]
    ├─ Decoder Block 2: Conv 3×3 → ReLU → Conv 3×3 → ReLU
    │
    └─ Output Conv 1×1 → 19 classes
```

Skip connections between encoder and decoder preserve spatial detail that would otherwise be lost during downsampling — important for small facial features like eyes and lips.

---

## Key Technical Decisions

### Resolution: 512×512 → 256×256
Training at the original 512×512 resolution caused instability and slow convergence (~80s/epoch). Dropping to 256×256 cut that to ~25s/epoch with no meaningful accuracy loss.

### Mask Encoding
CelebAMask-HQ stores masks as RGB images (one color per class). Standard cross-entropy requires integer class labels. Feeding raw RGB masks directly caused CUDA device-side assertion errors during early training. The fix: a deterministic color→ID mapping that converts every mask to a class index array, cached as `.npy` for fast loading.

```python
COLOR_TO_ID = {
    (0, 0, 0): 0,       # background
    (255, 255, 0): 1,   # skin
    (0, 255, 255): 2,   # left eyebrow
    # ... 16 more classes
}
```

### Loss Function: CE + Dice
Cross-entropy alone tends to over-optimize for large regions (skin, hair, background) and ignore small ones (eyes, lips, glasses). Adding Dice loss directly penalizes overlap errors on small classes.

```python
def combined_loss(logits, target):
    return ce_loss(logits, target) + dice_loss(logits, target)
```

### Memory Stability
Batch size 16 triggered OOM errors on the T4. Reducing to 8 and enabling `pin_memory=True` in the DataLoader stabilized training throughout all 50 epochs.

---

## Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| LR scheduler | Cosine Annealing |
| Epochs | 50 |
| Batch size | 8 |
| Input resolution | 256 × 256 |
| Augmentation | Random horizontal flip · Color jitter |

---

## Results

| Metric | Score |
|---|---|
| Pixel Accuracy | ~84% |
| Mean Dice (F1) | ~0.62 |

The model performs well on large regions (skin, hair, background). Smaller accessories like glasses and earrings are harder due to limited training examples — a data constraint, not an architectural one.

---

## Quick Start

```bash
pip install -r requirements.txt
```

Open `DL_final.ipynb` and run cells sequentially. The notebook covers the full pipeline: data loading, mask preprocessing, training, validation, and test inference.

The best model checkpoint is included as `model_best.pth`.

---

## Dataset

[CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) — a large-scale face dataset with pixel-level annotations for 19 facial components. This project used a fixed subset of 1,100 labeled images.

---

## Author

Simon Enkel
