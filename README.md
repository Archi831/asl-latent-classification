# ASL Latent Classification

Comparing two approaches to classifying 39 American Sign Language hand signs (A–Z, 0–9): a **direct CNN classifier** versus an **autoencoder + MLP pipeline** operating on compressed latent representations.

---

## Research Question

> Does learning a compact latent representation via an autoencoder produce a classifier that is equally accurate and more robust than a CNN trained end-to-end?

---

## Dataset

| Property | Value |
|---|---|
| Source | `data/asl-alphabet-numbers/asl-numbers-alphabet-dataset` |
| Classes | 39 — letters A–Z, digits 0–9 |
| Size | ~117,000 images (3,000 / class) |
| Input resolution | 64 × 64 grayscale, normalised to [0, 1] |
| Split (seed 67) | 70% train / 10% val / 20% test |

The fixed seed ensures all models operate on the exact same test set.

---

## Architecture Overview

```
Raw Image (1 × 64 × 64)
       │
       ├──────────────────────────────────────────┐
       │                                          │
       ▼                                          ▼
┌─────────────────┐                  ┌──────────────────────┐
│  Path A: CNN    │                  │  Path B: AE + MLP    │
│  (direct)       │                  │  (latent pipeline)   │
└────────┬────────┘                  └──────────┬───────────┘
         │                                      │
         │                           ┌──────────▼──────────┐
         │                           │  Encoder (CNN)       │
         │                           │  → 128-dim vector z  │
         │                           └──────────┬──────────┘
         │                                      │
         │                           ┌──────────▼──────────┐
         │                           │  MLP classifier      │
         │                           │  on latent z         │
         │                           └──────────┬──────────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        ▼
                 Predicted Sign
```

### CNN Classifier (`cnn_classifier/`)

Four ConvBlocks (32→64→128→256 filters) with MaxPool, then `AdaptiveAvgPool → Flatten → classifier head`. Three head variants are ablated: `shallow` (single linear), `standard` (256→ReLU→Dropout→39), `deep` (512→BN→ReLU→Dropout→256→ReLU→Dropout→39).

### Autoencoder (`autoencoder/`)

The encoder mirrors the CNN backbone. The decoder projects the bottleneck vector back through four bilinear upsampling stages — no skip connections. An auxiliary `ClassifierHead` is attached to `z` during training to make the latent space class-discriminative; it is discarded after training.

Loss: `0.6 × (0.5·MSE + 0.5·(1−SSIM)) + 0.4 × CrossEntropy (label smoothing 0.1)`

### MLP on Latents (`experiments/`)

The frozen encoder converts images to 128-dim L2-normalised vectors. An MLP (`128→256→128→64→39`) is trained on those vectors alone.

---

## Project Structure

```
asl-latent-classification/
├── autoencoder/
│   ├── autoencoder.py          # Encoder, Decoder, ClassifierHead, Autoencoder
│   ├── train_autoencoder.py    # Training loop + ablation logging
│   ├── save_latents.py         # Export latent vectors to .npy
│   ├── data_prep*.py           # Dataset loaders (clean + robustness variants)
│   └── models/                 # Saved checkpoints (ld64 / ld128 / ld256)
├── cnn_classifier/
│   ├── model.py                # CNNClassifier + head variants
│   ├── train.py                # Training + ablation logging
│   ├── evaluate.py             # Test-set evaluation
│   ├── evaluate_robustness.py  # Noise / brightness / resolution degradation
│   └── models/                 # Saved checkpoints (ab1–ab7)
├── experiments/                # MLP ablation scripts (Ab1, Ab2, Hp1, Hp2, M)
│   └── {bright,noise,resolution,dataset2}/  # Robustness variants
├── latents/                    # .npy latent vectors and labels
├── tsne/                       # t-SNE scripts and output plots
├── realworld/
│   ├── finetune_crosseval.py       # CNN cross-dataset fine-tuning
│   ├── finetune_aemplp_crosseval.py# AE+MLP cross-dataset evaluation
│   ├── evaluate_zeroshot_crosseval.py
│   ├── models/                     # Copies of best models for inference
│   └── results/                    # Per-class accuracy CSVs
├── data/
│   └── asl-alphabet-numbers/asl-numbers-alphabet-dataset/
├── docs/
│   ├── PROJECT_OVERVIEW.md         # Full design and results reference
│   ├── REALWORLD_EVAL_REPORT.md    # Cross-dataset evaluation report
│   └── *_results_overview.md       # Per-experiment per-class breakdowns
└── README.md
```

---

## Results

### Autoencoder latent-dim ablation

| Run | Latent dim | Best epoch | Val loss |
|---|---|---|---|
| ae_asl39_ld64 | 64 | 50 | 0.3034 |
| ae_asl39_ld128 | 128 | 50 | 0.3037 |
| ae_asl39_ld256 | 256 | 46 | **0.3029** |

All runs trained on ASL39 (39 classes, up to 50 epochs, early-stop patience 7).

### CNN ablation (in-distribution, ASL39 test set)

| Run | Head | Optimizer | Aug | LR sched | Val acc |
|---|---|---|---|---|---|
| ab1 baseline | standard | Adam | No | plateau | 99.94% |
| ab2 adamw | standard | AdamW | No | plateau | 99.90% |
| ab3 aug | standard | AdamW | Yes | plateau | 99.95% |
| ab4 aug+ls | standard | AdamW | Yes | plateau | **99.97%** |
| ab5 cosine | standard | AdamW | Yes | cosine | **99.97%** |
| ab6 shallow | shallow | AdamW | Yes | plateau | 99.95% |
| ab7 deep | deep | AdamW | Yes | plateau | trained |

**Best in-distribution:** ab4 / ab5 (99.97%). **Best cross-domain:** ab6 (shallow head).

### Real-world evaluation (zero-shot → cross-dataset fine-tuning)

| Experiment | Model | Ayuraj | Danrasband |
|---|---|:---:|:---:|
| Zero-shot | CNN ab6 | 28.1% | 16.2% |
| Zero-shot | AE+MLP | 27.5% | 5.0% |
| Fine-tune (danrasband→) | CNN ab5 | **40.3%** | — |
| Fine-tune (ayuraj→) | CNN ab6 | — | **34.5%** |
| Fine-tune (danrasband→) | AE+MLP full | 20.8% | — |
| Fine-tune (ayuraj→) | AE+MLP full | — | 10.0% |

Zero-shot ayuraj accuracy is dominated by digits (which transfer ~100%); letter recognition is near 0% for both models. After fine-tuning with as few as 30 images/class, accuracy roughly doubles. The CNN generalises substantially better than the AE+MLP pipeline under domain shift.

See [docs/REALWORLD_EVAL_REPORT.md](docs/REALWORLD_EVAL_REPORT.md) for full per-class breakdowns.

---

## Setup

```bash
pip install torch torchvision torchmetrics scikit-learn matplotlib seaborn tqdm pillow
```

The main dataset must be placed at:
```
data/asl-alphabet-numbers/asl-numbers-alphabet-dataset/
```

---

## Usage

### Train the autoencoder

```bash
cd autoencoder
python train_autoencoder.py --latent_dim 128
# Results logged to autoencoder/outputs/ae_ablation_results.csv
```

### Export latent vectors

```bash
python autoencoder/save_latents.py --latent_dim 128
# Saves latents/latents_asl39_ld128.npy and latents/labels_asl39_ld128.npy
```

### Train the CNN classifier

```bash
cd cnn_classifier
python train.py --head standard --optimizer adamw --augment --label_smoothing 0.05 --scheduler cosine
# Results logged to cnn_classifier/outputs/ablation_results_asl39.csv
```

### Evaluate CNN robustness

```bash
python cnn_classifier/evaluate_robustness.py
```

### Run t-SNE visualisation

```bash
python tsne/tsne_analysis_train_main.py
python tsne/tsne_analysis_test_main.py
# Plots saved to tsne/tsne_main_{train,test}.png
```

### Cross-dataset evaluation

```bash
# Zero-shot
python realworld/evaluate_zeroshot_crosseval.py

# CNN fine-tuning
python realworld/finetune_crosseval.py

# AE+MLP fine-tuning
python realworld/finetune_aemplp_crosseval.py --mode zeroshot
python realworld/finetune_aemplp_crosseval.py --mode mlp_only
python realworld/finetune_aemplp_crosseval.py --mode full
```

---

## Key Findings

1. **In-distribution accuracy is near-perfect for both pipelines.** The CNN hits 99.97% val accuracy. The MLP on latent vectors achieves competitive accuracy with far fewer parameters on the classification head.

2. **Domain shift is severe for letters, mild for digits.** Digit representations transfer across datasets out of the box; letter representations are tightly coupled to the training visual style (background, crop, lighting).

3. **The CNN generalises better under domain shift.** In every cross-dataset experiment, the CNN outperforms the AE+MLP pipeline — sometimes by large margins (danrasband zero-shot: 16% vs 5%).

4. **Shallower classifier heads generalise better.** The single-linear-layer ab6 consistently outperforms the deep-head ab7 on cross-domain tasks; the deep head overfits batch statistics to the fine-tuning distribution.

5. **Fine-tuning with minimal data is effective for the CNN.** Even 30 images/class roughly doubles cross-domain accuracy, confirming the backbone features are reusable.

---

## Team

| Person | Component |
|---|---|
| Lana | Autoencoder (architecture, training, latent export) |
| Lilia | MLP classifier on latent vectors + presentation |
| Maksym | t-SNE visualisation + robustness testing |
| Archie | CNN classifier + real-world evaluation + documentation |
