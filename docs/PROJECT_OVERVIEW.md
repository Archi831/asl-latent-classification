# ASL Latent Classification — Project Overview

**Branch:** `realworld-eval` | **Last updated:** 2026-04-25

## What Is This Project?

This project investigates two different approaches to classifying American Sign Language (ASL) hand signs from images, and compares how well each one performs — especially under difficult conditions.

The central question is:

> **Does learning a compact internal representation of images (via an autoencoder) produce a better or more robust classifier than a CNN that classifies images directly?**

There are **39 ASL signs** to recognize: letters A–Z plus digits 0–9. The dataset contains ~117,000 images (3,000 per class), all resized to 64×64 grayscale.

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

The latent space can be visualized using **t-SNE**, which squashes the 128 dimensions into 2D for plotting. If the autoencoder has learned meaningful representations, you should see 39 distinct clusters — one per sign class.

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
  → z  (N-dim latent vector, N ∈ {64, 128, 256})
```

The decoder mirrors this in reverse, using fully-connected layers to project `z` back to a `(256, 4, 4)` feature map, followed by four bilinear upsampling stages. There are **no skip connections** — the decoder receives only the bottleneck vector `z`, so all reconstruction must come from information the encoder chose to preserve there.

### Training

- **Loss function:** `0.6 × (0.5·MSE + 0.5·(1−SSIM)) + 0.4 × CrossEntropy` (reconstruction + auxiliary classification on latent z)
  - **MSE** penalizes pixel-level differences
  - **`(1−SSIM)`** penalizes structural/perceptual differences (SSIM = 1 is perfect, so `1−SSIM` is the loss)
  - **CrossEntropy** (label smoothing 0.1) on the auxiliary latent head encourages the latent space to be class-discriminative
- **Optimizer:** Adam, lr = 1e-3
- **LR Scheduler:** ReduceLROnPlateau (halves LR after 3 epochs without improvement)
- **Early stopping:** patience = 7 epochs
- **Epochs:** up to 50

### Latent-Dim Ablation Results (completed)

| Run | latent_dim | best_epoch | best_val_loss |
|---|---|---|---|
| ae_asl39_ld64  | 64  | 50 | 0.3034 |
| ae_asl39_ld128 | 128 | 50 | 0.3037 |
| ae_asl39_ld256 | 256 | **46** | **0.3029** |

**Best by val loss:** `ae_asl39_ld256` (0.3029). However, the AE+MLP pipeline and all cross-dataset experiments used the **ld128** encoder — the differences in val loss are marginal (0.3029 vs 0.3037) and ld128 was the working default throughout.

### What good results look like

- Reconstructed images should be recognizable as the same ASL sign as the originals
- Train and validation loss curves should converge without diverging (no overfitting)
- t-SNE of latent vectors should show 39 visually separable clusters

---

## Part 2 — MLP Classifier on Latent Vectors (Lilia)

### What it is

Once the autoencoder is trained, the encoder is frozen and used to convert every image into its latent vector. These vectors are saved as `.npy` files. A **Multi-Layer Perceptron (MLP)** — a simple stack of fully-connected layers — is then trained to classify the sign from the vector alone.

The MLP never sees pixel data. It only sees the abstract numbers that the encoder produced.

### Why this matters

If the autoencoder has learned a truly meaningful latent space, the MLP should be able to classify signs accurately with very little complexity. This would demonstrate that the encoder successfully extracted the discriminative structure of the data.

### Architecture

```
Input: N-dim latent vector (N ∈ {64, 128} depending on run)
  → Linear(N, 256) → ReLU
  → Linear(256, 128) → ReLU
  → Linear(128, 64)  → ReLU
  → Linear(64, num_classes)
  → logits
```

No dropout, batch norm, or weight decay — the input is already L2-normalised by the encoder, so the latent vectors are well-conditioned.

### Training

- **Loss:** CrossEntropyLoss (no label smoothing)
- **Optimizer:** Adam, lr = 5×10⁻⁴
- **Batch size:** 64
- **Epochs:** 20

### Ablation Experiments (29-class dataset)

The MLP architecture was **fixed** across all ablation runs. What varied between runs was the **encoder configuration** — i.e., which latent vectors were fed in. Each run loaded pre-saved `.npy` latent files produced by a differently-configured autoencoder.

Note: these ablation runs used a 29-class sign-language-mnist format dataset (letters A–Z + del/nothing/space), separate from the 39-class ASL39 dataset used by the autoencoder and CNN training.

| Run | What was varied | Latent source | Test acc |
|---|---|---|---|
| M (main) | Baseline | ld128 AE, MSE+SSIM+CE loss | **100%** |
| Ab1 | Feature reduction | Reduced-dim encoder | **100%** |
| Ab2 | Architecture/data tweak | Modified encoder config | **100%** |
| Hp1 | 64-dim latent + SSIM-only AE | ld64 AE, SSIM-only loss | **100%** |
| Hp2 | AE trained at lr = 5×10⁻⁴ | ld128 AE, lower lr | **100%** |

All five runs reached 100% accuracy. This demonstrates that the latent space is extremely linearly separable: once the encoder has learned a class-discriminative representation, even a simple 4-layer MLP classifies it perfectly. The result is robust to the encoder variant used.

### Robustness Experiments (39-class ASL39 dataset)

These runs test how gracefully the AE+MLP pipeline degrades when input images are **perturbed before encoding**. The same trained MLP classifier is used; only the input image quality changes.

| Perturbation | Parameters | AE+MLP Test acc | CNN Test acc |
|---|---|---|---|
| Brightness shift | `ColorJitter(brightness=0.5)` | 98.66% | **99.98%** |
| Gaussian noise | σ = 0.25 | 1.30% | 1.30% |
| Resolution drop | 64×64 → 16×16 → 64×64 bilinear | 22.29% | **29.02%** |
| Dataset2 (second distribution) | — | **99.90%** | — |

Both models collapse under heavy Gaussian noise (σ = 0.25) — neither model was exposed to noise during training, so no representation survives the corruption. Under resolution degradation, the CNN retains more accuracy (29.02% vs 22.29%). Both models remain high under brightness shift, with the CNN marginally ahead (99.98% vs 98.66%).

### What good results look like

- Accuracy of ≥ 99% on the in-distribution test set (demonstrated across all runs)
- Robustness: accuracy should drop by fewer than 1 pp under moderate perturbations
- Confusion matrix should be nearly diagonal; the few errors should fall on visually similar pairs (e.g., M/N, A/S, U/V)

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
  → [head] → logits (39 classes)
```

Three **head variants** are ablated:

| head | Architecture |
|---|---|
| `shallow` | `Linear(256, 39)` |
| `standard` | `Linear(256, 256) → ReLU → Dropout → Linear(256, 39)` |
| `deep` | `Linear(256, 512) → BN1d → ReLU → Dropout → Linear(512, 256) → ReLU → Dropout → Linear(256, 39)` |

The architecture mirrors the encoder intentionally, making the comparison fair — both models have similar capacity and see the same input.

### CNN Ablation Results (complete)

| # | Run name | head | optimizer | aug | ls | scheduler | val_acc | best_epoch |
|---|---|---|---|---|---|---|---|---|
| ab1 | asl39_ab1_baseline     | standard | adam  | No  | 0.00 | plateau | 99.94% | 19/26 |
| ab2 | asl39_ab2_adamw        | standard | adamw | No  | 0.00 | plateau | 99.90% | 27/34 |
| ab3 | asl39_ab3_aug          | standard | adamw | Yes | 0.00 | plateau | 99.95% | 34/41 |
| ab4 | asl39_ab4_aug_ls       | standard | adamw | Yes | 0.05 | plateau | **99.97%** | 50/50 |
| ab5 | asl39_ab5_cosine       | standard | adamw | Yes | 0.05 | cosine  | **99.97%** | 45/50 |
| ab6 | asl39_ab6_head_shallow | shallow  | adamw | Yes | 0.05 | plateau | 99.95% | 50/50 |
| ab7 | asl39_ab7_head_deep    | deep     | adamw | Yes | 0.05 | plateau | trained | — |

All three head variants (ab5/ab6/ab7) are used in real-world evaluation. **Best in-distribution model:** `asl39_ab5_cosine` (99.97% val acc). **Best cross-domain model:** `ab6` (shallow head, consistently best on external datasets).

Results logged to `cnn_classifier/outputs/ablation_results_asl39.csv`.

### Role in the project

This is the **baseline**. It answers: "How well does a standard approach do?" The CNN trains directly on labels and is optimized for classification from the start, which typically gives it an advantage in raw accuracy. The interesting comparison is whether the MLP+latent pipeline can match it — and whether either approach is more **robust** to noisy inputs.

### What good results look like

- High test accuracy (expected to be the accuracy ceiling of the project)
- Clean loss curves with no overfitting
- Confusion matrix similar in structure to the MLP's (same hard cases)

---

## Part 4 — t-SNE Visualization & Robustness Testing (Maksym)

---

## Part 5 — Real-World Generalization (Archie)

### What it is

A suite of experiments evaluating how well the trained models transfer to external ASL datasets with different signers, backgrounds, and lighting — none of which appeared in ASL39 training.

### Datasets used

| Dataset | Classes | Images/class | Character |
|---|---|---|---|
| ayuraj/asl-dataset | A–Z + 0–9 (36) | ~70 | Controlled lighting, uniform background, hand-cropped |
| danrasband/asl-alphabet-test | A–Z (26) | 30 | Real-world footage, single subject, natural background |

debashishsau was investigated and excluded — it contains images from the ASL39 training corpus, ayuraj, and danrasband, and consists almost entirely of synthetic augmentations. See `docs/debashishsau_exclusion_note.md`. No multi-source fine-tuning experiments were conducted.

### Results summary

**Zero-shot (no adaptation):**

| Model | Ayuraj | Danrasband |
|---|---|---|
| CNN ab5 (standard) | 27.5% | 14.7% |
| CNN ab6 (shallow) | **28.1%** | **16.2%** |
| CNN ab7 (deep) | 27.5% | 13.1% |
| AE+MLP | 27.5% | 5.0% |

Ayuraj digits score ~97–100% across all models; every letter class scores ~0%. The 27–28% headline is almost entirely explained by digit images. On danrasband (letters only), the CNN holds at 13–16%; the AE+MLP collapses to 5%.

**Cross-dataset fine-tuning (CNN):**

| Fine-tuned on | Tested on | Best accuracy (model) |
|---|---|---|
| danrasband | ayuraj | 40.3% (ab5) |
| ayuraj | danrasband | **34.5%** (ab6) |

Fine-tuning with as few as 30 images/class roughly doubles cross-domain accuracy. The shallow head (ab6) generalises best; the deep head (ab7) overfits the fine-tuning distribution. The AE+MLP pipeline reaches only 20.8% / 10.0% under the same conditions.

### Key findings

- Domain shift is severe for letters, mild for digits.
- The CNN adapts better than the AE+MLP under limited fine-tuning data.
- Shallower classifier heads generalise better cross-domain.

See `docs/REALWORLD_EVAL_REPORT.md` for complete per-class breakdowns and methodology.

### t-SNE

**t-SNE** (t-distributed Stochastic Neighbor Embedding) is a dimensionality reduction technique that takes the 128-dim latent vectors and projects them into 2D while preserving local structure. The result is a scatter plot where:

- Each point is one image
- Each color is one of the 29 sign classes
- **Clusters** indicate that the encoder learned to map similar signs to nearby points in latent space

If the scatter plot shows 39 tight, well-separated clusters, the autoencoder has successfully learned a meaningful structure of the data. Overlapping clusters indicate signs that the encoder cannot distinguish.

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

- **Source:** `data/asl-alphabet-numbers/asl-numbers-alphabet-dataset`
- **Size:** ~117,000 images (3,000 per class)
- **Classes:** 39 — A through Z, digits 0–9
- **Source resolution:** 224×224
- **Used resolution:** 64×64 grayscale (normalized to [0, 1])
- **Split (seed=67):** 70% train / 10% val / 20% test

The fixed seed ensures all team members work with the exact same test set, making results directly comparable.

---

## What Does Success Look Like?

| Component | Target |
|---|---|
| Autoencoder reconstructions | Visually recognizable signs at convergence |
| Autoencoder val loss (combined) | ~0.30 (achieved: ld256=0.3029) |
| MLP accuracy on latent vectors | > 90% on test set |
| CNN baseline accuracy | > 90% on test set (ab1 baseline: **99.94% val acc**) |
| t-SNE | 39 visually separable clusters |
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
| Lana | Autoencoder (encoder + decoder + training + latent export) | Done — AE ablation complete (best: ld256, val_loss=0.3029) |
| Archie | CNN classifier + real-world evaluation + project documentation | Done |
| Lilia | MLP on latent vectors + presentation | Done |
| Maksym | t-SNE visualization + robustness testing | Done |

## Current Status (2026-04-25)

- [x] AE ablation complete — best: ld256, val_loss=0.3029
- [x] CNN ablation ab1–ab7 complete — best in-distribution: ab5-cosine (99.97% val acc)
- [x] AE+MLP pipeline trained and evaluated
- [x] t-SNE visualization complete
- [x] Robustness testing (noise, brightness, resolution) complete
- [x] Zero-shot evaluation on ayuraj and danrasband — CNN (ab5/ab6/ab7) and AE+MLP
- [x] Cross-dataset fine-tuning (danrasband↔ayuraj) — CNN (ab5/ab6/ab7) and AE+MLP
- [x] debashishsau investigated and excluded (training-set contamination + synthetic image dominance)
- [x] Real-world evaluation report complete (`docs/REALWORLD_EVAL_REPORT.md`)
