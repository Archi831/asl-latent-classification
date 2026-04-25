# ASL Latent Classification — Project Overview

## What Is This Project?

This project investigates two different approaches to classifying American Sign Language (ASL) hand signs from images, and compares how well each one performs — especially under difficult conditions.

The central question is:

> **Does learning a compact internal representation of images (via an autoencoder) produce a better or more robust classifier than a CNN that classifies images directly?**

There are 29 ASL signs to recognize: letters A–Z plus Space, Delete, and Nothing. The dataset contains ~87,000 images (3,000 per class), all resized to 64×64 grayscale.

---

## The Core Idea: Two Paths to Classification

```
                        ┌─────────────────────────────────┐
                        │          Raw Image (64x64)       │
                        └──────────────┬──────────────────┘
                                       │
               ┌───────────────────────┼───────────────────────┐
               │                       │                       │
               ▼                       ▼                       │
    ┌─────────────────┐     ┌─────────────────────┐           │
    │  Path A: Direct │     │  Path B: Latent      │           │
    │  CNN Classifier │     │  Space Pipeline      │           │
    └────────┬────────┘     └──────────┬──────────┘           │
             │                         │                       │
             │              ┌──────────▼──────────┐           │
             │              │  Encoder (CNN)       │           │
             │              │  compresses image    │           │
             │              │  → 128-dim vector    │           │
             │              └──────────┬──────────┘           │
             │                         │                       │
             │              ┌──────────▼──────────┐           │
             │              │  MLP Classifier      │           │
             │              │  classifies vector   │           │
             │              └──────────┬──────────┘           │
             │                         │                       │
             └───────────────┬─────────┘                       │
                             │                                 │
                             ▼                                 │
                      Predicted Sign                          │
                      (A–Z, Space, etc.)                      │
                                                              │
                    t-SNE Visualization ◄─────────────────────┘
```

---

## What Is a Latent Space?

When a neural network processes an image, it transforms the raw pixel data (64×64 = 4,096 numbers) into increasingly abstract internal representations. The **latent space** is a compressed version of this — a small, dense vector that captures the most essential features of the image.

In this project, the encoder compresses each 64×64 image into a **128-dimensional vector** (128 numbers). This is a ~32× reduction in size.

Think of it this way: instead of storing the full painting, you store a list of 128 style descriptors. If the model has learned well, signs that look similar (e.g., A and S) will have vectors that are close together in this 128-dimensional space, while signs that look different (e.g., B and C) will be far apart.

The latent space can be visualized using **t-SNE**, which squashes the 128 dimensions into 2D for plotting. If the autoencoder has learned meaningful representations, you should see 29 distinct clusters — one per sign class.

---

## Part 1 — Autoencoder (Lana)

### What it is

An **autoencoder** is a neural network trained to reconstruct its own input. It has two parts:

- **Encoder** — compresses the image into a latent vector `z` (128 numbers)
- **Decoder** — reconstructs the original image from `z`

The network is forced to learn only what is essential about the image, because all information must pass through the bottleneck of 128 numbers.

### Architecture

The encoder is a convolutional network with 4 downsampling stages:

```
Input (1, 64, 64)
  → ConvBlock (32 filters)  + MaxPool  → (32, 32, 32)
  → ConvBlock (64 filters)  + MaxPool  → (64, 16, 16)
  → ConvBlock (128 filters) + MaxPool  → (128, 8, 8)
  → ConvBlock (256 filters) + MaxPool  → (256, 4, 4)
  → Flatten → FC(4096→512) → Dropout(0.3) → FC(512→128)
  → z  (128-dim latent vector)
```

The decoder mirrors this in reverse, using bilinear upsampling and **skip connections** (U-Net style). Skip connections pass feature maps from the encoder directly to the matching decoder stage, which helps the decoder recover fine spatial details that would otherwise be lost in the bottleneck.

### Training

- **Loss function:** Combined MSE + SSIM (50% each)
  - **MSE** (Mean Squared Error) penalizes pixel-level differences
  - **SSIM** (Structural Similarity Index) additionally penalizes differences in local contrast, luminance, and structure — it aligns better with how humans perceive image quality
- **Optimizer:** Adam, lr = 1e-3
- **LR Scheduler:** ReduceLROnPlateau (halves LR after 3 epochs without improvement)
- **Early stopping:** patience = 7 epochs
- **Epochs:** up to 50

### What good results look like

- Reconstructed images should be recognizable as the same ASL sign as the originals
- Train and validation loss curves should converge without diverging (no overfitting)
- t-SNE of latent vectors should show 29 visually separable clusters

---

## Part 2 — MLP Classifier on Latent Vectors (Lilia)

### What it is

Once the autoencoder is trained, the encoder is frozen and used to convert every image in the dataset into its 128-dim latent vector. These vectors are saved as `.npy` files. A **Multi-Layer Perceptron (MLP)** — a simple stack of fully-connected layers — is then trained to classify the sign from the vector alone.

The MLP never sees pixel data. It only sees the 128 abstract numbers that the encoder produced.

### Why this matters

If the autoencoder has learned a truly meaningful latent space, the MLP should be able to classify signs accurately with very little complexity (just a few dense layers). This would demonstrate that the encoder successfully extracted the discriminative structure of the data.

### What good results look like

- Accuracy of 90%+ on the test set is a reasonable target given 29 classes
- Confusion matrix should show most confusion between visually similar signs (e.g., M/N, A/S, U/V/W)
- Signs like "Nothing" and "Space" may be inherently harder to distinguish

---

## Part 3 — CNN Classifier (Archie)

### What it is

A standard convolutional neural network that takes a raw 64×64 image and directly outputs a class prediction. No autoencoder, no latent space — just end-to-end supervised classification.

### Architecture

```
Input (1, 64, 64)
  → ConvBlock (32)  + MaxPool → (32, 32, 32)
  → ConvBlock (64)  + MaxPool → (64, 16, 16)
  → ConvBlock (128) + MaxPool → (128, 8, 8)
  → ConvBlock (256) + MaxPool → (256, 4, 4)
  → AdaptiveAvgPool → Flatten → (256)
  → FC(256→256) + ReLU + Dropout(0.5) → FC(256→29)
  → logits (29 classes)
```

The architecture mirrors the encoder intentionally, making the comparison fair — both models have similar capacity and see the same input.

### Role in the project

This is the **baseline**. It answers: "How well does a standard approach do?" The CNN trains directly on labels and is optimized for classification from the start, which typically gives it an advantage in raw accuracy. The interesting comparison is whether the MLP+latent pipeline can match it — and whether either approach is more **robust** to noisy inputs.

### What good results look like

- High test accuracy (expected to be the accuracy ceiling of the project)
- Clean loss curves with no overfitting
- Confusion matrix similar in structure to the MLP's (same hard cases)

---

## Part 4 — t-SNE Visualization & Robustness Testing (Maksym)

### t-SNE

**t-SNE** (t-distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that takes the 128-dim latent vectors and projects them into 2D while preserving local structure. The result is a scatter plot where:

- Each point is one image
- Each color is one of the 29 sign classes
- **Clusters** indicate that the encoder learned to map similar signs to nearby points in latent space

If the scatter plot shows 29 tight, well-separated clusters, the autoencoder has successfully learned a meaningful structure of the data. Overlapping clusters indicate signs that the encoder cannot distinguish.

### Robustness testing

Both models (CNN and MLP) are tested on **degraded versions** of the test set:

| Perturbation | What it simulates |
|---|---|
| Gaussian noise | Camera sensor noise, low-light conditions |
| Brightness shift | Over/underexposure, different lighting environments |
| Resolution drop | Low-resolution cameras, distant signers |

The test measures how much each model's accuracy drops as the degradation increases. A robust model degrades gracefully; a brittle model collapses quickly.

This is one of the most practically relevant experiments: real-world ASL recognition systems must work in imperfect conditions.

---

## Dataset

- **Source:** [ASL Alphabet on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) (grassknoted)
- **Size:** ~87,000 images (3,000 per class)
- **Classes:** 29 — A through Z, Space, Delete, Nothing
- **Original resolution:** 200×200 RGB
- **Used resolution:** 64×64 grayscale (normalized to [0, 1])
- **Split (seed=67):** 70% train / 10% val / 20% test (~60,900 / ~8,700 / ~17,400 images)

The fixed seed ensures all team members work with the exact same test set of 17,400 images, making results directly comparable.

---

## What Does Success Look Like?

| Component | Target |
|---|---|
| Autoencoder reconstructions | Visually recognizable signs at convergence |
| Autoencoder test loss (MSE+SSIM) | < 0.05 combined loss |
| MLP accuracy on latent vectors | > 90% on test set |
| CNN baseline accuracy | > 90% on test set |
| t-SNE | 29 visually separable clusters |
| Robustness | Graceful degradation; identify which model is more brittle |

The most interesting result is not which model wins in accuracy, but **why** — and whether the latent space representation confers any advantage in robustness or interpretability.

---

## Why Is This Project Interesting?

Direct CNN classification is a solved problem for clean, controlled datasets like this one. The interesting research question is about **representation learning**:

1. **Interpretability** — the latent space can be visualized and inspected. The CNN's internal state is opaque; the 128-dim vector is at least inspectable via t-SNE.

2. **Modularity** — once the encoder is trained, it can be reused. The MLP classifier can be swapped out or retrained independently without touching the encoder. This is the foundation of **transfer learning**.

3. **Robustness** — a model that has learned a compact, abstract representation may be less sensitive to surface-level noise in pixels. This is testable, which is exactly what the robustness experiments measure.

4. **Dimensionality and generalization** — the MLP classifies from 128 numbers instead of 4,096 pixels. Fewer input dimensions often means less capacity for overfitting to irrelevant variation.

---

## Team Responsibilities

| Person | Component | Status |
|---|---|---|
| Lana | Autoencoder (encoder + decoder + training + latent export) | Done |
| Archie | CNN classifier + project documentation | In progress |
| Lilia | MLP on latent vectors + presentation | TODO |
| Maksym | t-SNE visualization + robustness testing | TODO |
