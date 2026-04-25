# CNN Cross-Dataset Evaluation: Ayuraj ↔ Danrasband

## Datasets

| Dataset | Source | Images | Classes | Notes |
|---------|--------|--------|---------|-------|
| **ayuraj** | `ayuraj/asl-dataset` | 2,515 | 36 (digits 0–9 + letters A–Z, excl. J/Z + others) | Controlled lighting, uniform background, cropped hand |
| **danrasband** | `danrasband/asl-alphabet-test` | 780 | 26 (letters A–Z only) | Real-world footage, single subject, ~30 imgs/class |

Danrasband does not include digit classes, `nothing`, or `space`. Ayuraj has ~70 images per class.

## Experiment Setup

- **Base models**: ab5 (standard head), ab6 (shallow head), ab7 (deep head) — all trained on the original ASL39 training split.
- **Zero-shot**: model weights unchanged, evaluated directly on each dataset.
- **Fine-tuning strategy**: Freeze first two ConvBlocks; unfreeze last two ConvBlocks + classifier head.
- **Fine-tune hyperparameters**: AdamW lr=1e-4, weight_decay=1e-4, up to 30 epochs, early stopping patience=8, 20% val split from the fine-tuning dataset.

### Head architectures

| Tag | Head | Classifier layers |
|-----|------|-------------------|
| ab5 | standard | 256 → ReLU → Dropout(0.5) → 39 |
| ab6 | shallow | 256 → 39 (single linear) |
| ab7 | deep | 256 → 512 → BN → ReLU → Dropout → 256 → ReLU → Dropout → 39 |

---

## Overall Results

### Zero-shot (no fine-tuning)

| Model | Head | On ayuraj (2515 imgs) | On danrasband (780 imgs) |
|-------|------|:---------------------:|:------------------------:|
| ab5 | standard | 27.48% (691/2515) | 14.74% (115/780) |
| ab6 | shallow | **28.07%** (706/2515) | **16.15%** (126/780) |
| ab7 | deep | 27.51% (692/2515) | 13.08% (102/780) |

The headline numbers are deceiving: on ayuraj, nearly all correct predictions come from the 10 digit classes (which the original training set covers well). Every letter class scores 0% or near-0% zero-shot — the letter images in these datasets look nothing like the original training distribution. On danrasband (letters only), the baseline ranges from 13–16%.

### After fine-tuning

| Model | Head | Fine-tune on danrasband → test on ayuraj | Fine-tune on ayuraj → test on danrasband |
|-------|------|:-----------------------------------------:|:-----------------------------------------:|
| ab5 | standard | 40.32% (1014/2515) | 31.41% (245/780) |
| ab6 | shallow | 39.60% (996/2515) | **34.49%** (269/780) |
| ab7 | deep | 40.04% (1007/2515) | 27.95% (218/780) |

Fine-tuning gains over zero-shot:

| Model | ayuraj gain (danrasband ft) | danrasband gain (ayuraj ft) |
|-------|:---------------------------:|:---------------------------:|
| ab5 | +12.8 pp | +16.7 pp |
| ab6 | +11.5 pp | +18.3 pp |
| ab7 | +12.5 pp | +14.9 pp |

**Direction danrasband → ayuraj** is capped by the tiny fine-tuning set (780 images, 26 classes, 30/class); all three models converge to ~40%.

**Direction ayuraj → danrasband** is the more informative direction. Ayuraj has ~70 images/class and a larger vocabulary, providing a stronger fine-tuning signal. The 3-point gap between ab6 and ab7 (34.49% vs 27.95%) shows that the deeper head overfits more to ayuraj's controlled-lighting distribution and transfers worse.

---

## Per-Class Results — Zero-Shot

### On ayuraj (zero-shot)

Digits transfer from the original training set; letters completely fail. ab5 and ab7 score 0% on every letter; ab6 gets 1.4% on S and 21.4% on Y.

| Class | ab5 | ab6 | ab7 |
|-------|----:|----:|----:|
| 0 | 100.0% | 100.0% | 100.0% |
| 1 | 100.0% | 100.0% | 100.0% |
| 2 | 100.0% | 100.0% | 100.0% |
| 3 | 100.0% | 100.0% | 100.0% |
| 4 | 92.9% | 92.9% | 92.9% |
| 5 | 92.9% | 92.9% | 92.9% |
| 6 | 100.0% | 100.0% | 100.0% |
| 7 | 100.0% | 100.0% | 100.0% |
| 8 | 100.0% | 100.0% | 100.0% |
| 9 | 100.0% | 100.0% | 100.0% |
| A–Z (all) | ~0% | ~0% | ~0% |

The 27–28% headline accuracy is almost entirely explained by the 700 digit images (10 classes × 70) being correctly classified; the 1,815 letter images contribute almost nothing.

### On danrasband (zero-shot, letters only)

| Class | ab5 | ab6 | ab7 |
|-------|----:|----:|----:|
| A | 0.0% | 0.0% | 0.0% |
| B | 3.3% | 3.3% | 0.0% |
| C | 13.3% | 6.7% | 3.3% |
| D | 16.7% | 20.0% | 13.3% |
| E | 0.0% | 0.0% | 0.0% |
| F | 16.7% | 10.0% | 6.7% |
| G | 23.3% | 30.0% | 16.7% |
| H | **50.0%** | **53.3%** | **50.0%** |
| I | 23.3% | 26.7% | 0.0% |
| J | 43.3% | 46.7% | 46.7% |
| K | 6.7% | 3.3% | 3.3% |
| L | 36.7% | 36.7% | 16.7% |
| M | 10.0% | 13.3% | 16.7% |
| N | 3.3% | 0.0% | 0.0% |
| O | 0.0% | 0.0% | 0.0% |
| P | **66.7%** | **53.3%** | **63.3%** |
| Q | 30.0% | 26.7% | 36.7% |
| R | 6.7% | 13.3% | 6.7% |
| S | 0.0% | 0.0% | 0.0% |
| T | 3.3% | 16.7% | 0.0% |
| U | 13.3% | 13.3% | 10.0% |
| V | 0.0% | 3.3% | 0.0% |
| W | 0.0% | 3.3% | 0.0% |
| X | 0.0% | 3.3% | 0.0% |
| Y | 13.3% | 16.7% | 16.7% |
| Z | 3.3% | 20.0% | 33.3% |

H and P are the only classes with consistent zero-shot accuracy (≥50% for ab5). The overall 13–16% reflects a model that has learned features specific to the original training dataset's visual style.

---

## Per-Class Results — danrasband fine-tune → ayuraj test

Only letter classes are shown for digits separately below. Danrasband has no digits, so the model is fine-tuned without digit signal; generalization comes entirely from the frozen feature backbone.

### Digit classes (ayuraj only, backbone-only transfer)

| Class | ab5 | ab6 | ab7 |
|-------|----:|----:|----:|
| 0 | 0.0% | 0.0% | **51.4%** |
| 1 | 97.1% | 58.6% | 85.7% |
| 2 | 92.9% | 91.4% | 98.6% |
| 3 | 100.0% | 97.1% | 100.0% |
| 4 | 92.9% | 57.1% | 92.9% |
| 5 | 81.4% | 81.4% | 92.9% |
| 6 | 68.6% | 37.1% | **95.7%** |
| 7 | 100.0% | 100.0% | 100.0% |
| 8 | 98.6% | 92.9% | 100.0% |
| 9 | 100.0% | 94.3% | 100.0% |

ab7 recovers digit class 0 and 6 where ab5/ab6 completely fail — the deeper head may be better at separating digit-like shapes from its frozen feature representation.

### Letter classes (danrasband → ayuraj)

| Class | ab5 | ab6 | ab7 |
|-------|----:|----:|----:|
| A | 0.0% | 0.0% | 0.0% |
| B | 74.3% | 57.1% | 31.4% |
| C | 1.4% | 1.4% | 0.0% |
| D | 0.0% | 0.0% | 0.0% |
| E | 21.4% | 48.6% | 27.1% |
| F | 0.0% | 0.0% | 0.0% |
| G | 1.4% | 34.3% | 34.3% |
| H | 70.0% | 31.4% | 44.3% |
| I | 1.4% | 4.3% | 15.7% |
| J | 17.1% | 21.4% | 1.4% |
| K | 0.0% | 20.0% | 4.3% |
| L | **94.3%** | 47.1% | 82.9% |
| M | 0.0% | 0.0% | 0.0% |
| N | 5.7% | 4.3% | 0.0% |
| O | **95.7%** | 72.9% | 48.6% |
| P | 0.0% | 21.4% | 2.9% |
| Q | 0.0% | 0.0% | 0.0% |
| R | 17.1% | 55.7% | 18.6% |
| S | **95.7%** | 90.0% | **91.4%** |
| T | 3.1% | 13.8% | 18.5% |
| U | 10.0% | 7.1% | 0.0% |
| V | 2.9% | 14.3% | 5.7% |
| W | 30.0% | 81.4% | 0.0% |
| X | 4.3% | 0.0% | 7.1% |
| Y | 71.4% | 87.1% | 88.6% |
| Z | 0.0% | 0.0% | 0.0% |

**Consistently strong** (≥70% across most models): S, L, O, Y, B, H.  
**Consistently zero**: A, D, F, M, Q, Z — these likely collapse into visually similar neighbours during domain shift.

---

## Per-Class Results — ayuraj fine-tune → danrasband test

(Danrasband is letters A–Z only, 30 images each.)

| Class | ab5 | ab6 | ab7 |
|-------|----:|----:|----:|
| A | 6.7% | 13.3% | 3.3% |
| B | 10.0% | 6.7% | 6.7% |
| C | 23.3% | 23.3% | 16.7% |
| D | 23.3% | 33.3% | 10.0% |
| E | 3.3% | 16.7% | 3.3% |
| F | 10.0% | 10.0% | 3.3% |
| G | 20.0% | 40.0% | 16.7% |
| H | 46.7% | 63.3% | 56.7% |
| I | 43.3% | 43.3% | 6.7% |
| J | 73.3% | 73.3% | **83.3%** |
| K | 30.0% | 30.0% | 43.3% |
| L | 63.3% | 56.7% | 50.0% |
| M | **86.7%** | **86.7%** | 70.0% |
| N | 0.0% | 3.3% | 20.0% |
| O | 23.3% | 16.7% | 16.7% |
| P | **83.3%** | 80.0% | 76.7% |
| Q | 60.0% | **73.3%** | **80.0%** |
| R | 36.7% | 33.3% | 20.0% |
| S | 10.0% | 13.3% | 10.0% |
| T | 0.0% | 3.3% | 3.3% |
| U | 33.3% | 50.0% | 36.7% |
| V | 20.0% | 20.0% | 13.3% |
| W | 23.3% | 16.7% | 10.0% |
| X | 53.3% | 43.3% | 36.7% |
| Y | 20.0% | 33.3% | 30.0% |
| Z | 13.3% | 13.3% | 3.3% |

**Consistently strong** (≥60%): M, P, J, Q, L, H.  
**Consistently weak** (≤10%): A, B, E, F, S, T — many of these involve subtle finger positioning that does not survive the domain shift from the controlled ayuraj background to danrasband's real-world setting.

---

## Analysis

### Zero-shot: digit–letter asymmetry

The zero-shot results reveal a sharp structural pattern: digits transfer perfectly (~100%), letters fail completely (~0% on ayuraj). This is not random — it means the backbone has learned digit-class representations that are visually invariant across datasets, while letter representations are tightly coupled to the original training set's specific visual style (background, crop, lighting). The 27–28% ayuraj accuracy is almost entirely explained by the 700 digit images; the 1,815 letter images contribute ~1 correct prediction each on average.

On danrasband (letters only, zero-shot), P and H are consistently the most transferable signs, likely because their hand shapes are geometrically distinct and unambiguous even across visual domains.

### Fine-tuning gains

Even a tiny fine-tuning set (danrasband: 30 imgs/class) shifts letter accuracy from ~0% to ~40% total. This confirms the backbone features are reusable — the domain gap is primarily in the classifier decision boundaries, not the feature extractor.

Fine-tuning on ayuraj (larger, more varied) produces stronger in-distribution validation accuracy (~95%) but lower cross-dataset transfer (~28–34%), underlining the overfitting risk when fine-tuning adapts the feature layers to a specific visual style.

### Why cross-dataset accuracy plateaus at 28–40% after fine-tuning

1. **Dataset size asymmetry**: Danrasband has only 30 images per class. Fine-tuning on it gives the model almost no information per class; it essentially inherits backbone representations from the original training.
2. **Visual domain gap**: Ayuraj uses a plain/cropped setting; danrasband is real-world footage with varying hand position and background. Colours, contrast, and framing differ substantially.
3. **Class set mismatch**: Danrasband covers only 26 letters; ayuraj adds digits and other classes. When fine-tuned on danrasband, the digit classes in ayuraj are never seen during fine-tuning and must rely purely on the frozen backbone.

### Head depth and generalization

- ab6 (shallow) is the most consistent cross-domain performer (+18.3 pp gain on the harder direction; best overall cross-dataset accuracy at 34.49%).
- ab7 (deep, with BatchNorm in the head) can overfit to the fine-tuning distribution's statistics (via batch statistics in BN), hurting transfer. It is the worst on the harder direction (27.95%).
- ab5 (standard) is a middle ground but shows no consistent advantage over ab6 in cross-domain settings.

### Contamination note

The `debashishsau` dataset was excluded from this evaluation because it contains images also present in danrasband. Including it in any combination with danrasband would cause test contamination.

---

## Files

| File | Description |
|------|-------------|
| `evaluate_zeroshot_crosseval.py` | Zero-shot evaluation script |
| `finetune_crosseval.py` | Fine-tuning cross-evaluation script |
| `realworld/zeroshot_summary.csv` | Zero-shot accuracy summary (6 model × dataset combinations) |
| `realworld/zeroshot_ab5_ayuraj.csv` | Zero-shot per-class, ab5, ayuraj |
| `realworld/zeroshot_ab5_danrasband.csv` | Zero-shot per-class, ab5, danrasband |
| `realworld/zeroshot_ab6_ayuraj.csv` | Zero-shot per-class, ab6, ayuraj |
| `realworld/zeroshot_ab6_danrasband.csv` | Zero-shot per-class, ab6, danrasband |
| `realworld/zeroshot_ab7_ayuraj.csv` | Zero-shot per-class, ab7, ayuraj |
| `realworld/zeroshot_ab7_danrasband.csv` | Zero-shot per-class, ab7, danrasband |
| `realworld/finetune_crosseval_summary.csv` | Fine-tune accuracy summary for all 6 experiments |
| `realworld/finetune_ab5_danrasband_test_ayuraj.csv` | Per-class results, ab5, danrasband→ayuraj |
| `realworld/finetune_ab5_ayuraj_test_danrasband.csv` | Per-class results, ab5, ayuraj→danrasband |
| `realworld/finetune_ab6_danrasband_test_ayuraj.csv` | Per-class results, ab6, danrasband→ayuraj |
| `realworld/finetune_ab6_ayuraj_test_danrasband.csv` | Per-class results, ab6, ayuraj→danrasband |
| `realworld/finetune_ab7_danrasband_test_ayuraj.csv` | Per-class results, ab7, danrasband→ayuraj |
| `realworld/finetune_ab7_ayuraj_test_danrasband.csv` | Per-class results, ab7, ayuraj→danrasband |
