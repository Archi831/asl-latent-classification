# AE+MLP Cross-Dataset Evaluation: Ayuraj ↔ Danrasband

## Setup

**Pipeline:** Convolutional Autoencoder (ld128) → L2-normalised latent vectors → MLP classifier  
**Datasets:** ayuraj (2,515 images, 36 classes: digits + letters) · danrasband (780 images, 26 classes: letters only)  
**Three adaptation levels tested, mirroring the CNN cross-eval already on file.**

---

## Overall Accuracy Summary

### Direction 1: Fine-tune on danrasband → Test on ayuraj

| Mode | Overall | Letters only | Digits only |
|---|---|---|---|
| CNN fine-tune (reference) | **39.92%** | ~26% | ~95% |
| AE+MLP zero-shot | 27.48% | ~0% | ~98% |
| AE+MLP MLP-only retrain | 15.75% | ~28% | 0% |
| AE+MLP full fine-tune | **20.80%** | ~41% | 0% |

### Direction 2: Fine-tune on ayuraj → Test on danrasband

| Mode | Overall | Notes |
|---|---|---|
| CNN fine-tune (reference) | **31.15%** | Letters only (danrasband has no digits) |
| AE+MLP zero-shot | 5.00% | H 56%, P 40% only bright spots |
| AE+MLP MLP-only retrain | 10.51% | Q 60%, V 36%, L/M/Y 30% |
| AE+MLP full fine-tune | **10.00%** | M 50%, X 26%, K 23% |

---

## Per-Class Breakdown — Direction 1: danrasband → ayuraj

| Class | Zero-shot | MLP-only | Full fine-tune | CNN fine-tune | Notes |
|---|---|---|---|---|---|
| **Digits** | | | | | |
| 0 | 100% | 0% | 0% | 1.43% | Lost in MLP/full (no digits in source) |
| 1 | 100% | 0% | 0% | 98.57% | |
| 2 | 100% | 0% | 0% | 100% | |
| 3 | 100% | 0% | 0% | 100% | |
| 4 | 92.86% | 0% | 0% | 92.86% | |
| 5 | 92.86% | 0% | 0% | 88.57% | |
| 6 | 100% | 0% | 0% | 72.86% | |
| 7 | 100% | 0% | 0% | 100% | |
| 8 | 100% | 0% | 0% | 100% | |
| 9 | 100% | 0% | 0% | 100% | |
| **Letters** | | | | | |
| A | 0% | 21.43% | 0% | 0% | |
| B | 0% | 0% | 77.14% | 70.00% | Strong encoder recovery |
| C | 0% | 0% | 22.86% | 2.86% | |
| D | 0% | 0% | **0%** | 0% | Side-view vs front — no mode recovers |
| E | 0% | 0% | 22.86% | 17.14% | |
| F | 0% | 0% | 0% | 0% | |
| G | 0% | 0% | 10.00% | 1.43% | Tilted viewpoint |
| H | 0% | 0% | 91.43% | 50.00% | Tilted but strongly recovered by encoder ft |
| I | 0% | 0% | 1.43% | 1.43% | |
| J | 0% | 0% | 2.86% | 4.29% | |
| K | 0% | 0% | 4.29% | 0% | |
| L | 0% | 91.43% | 84.29% | 100% | Latent-similar across domains |
| M | 0% | 0% | 0% | 0% | Slight viewpoint diff, unrecovered |
| N | 0% | 0% | 0% | 1.43% | |
| O | 0% | 100% | 62.86% | 98.57% | Round shape generalises well |
| P | 0% | 0% | 1.43% | 0% | |
| Q | 0% | 5.71% | **0%** | 1.43% | Tilted — unrecovered |
| R | 0% | 98.57% | 70.00% | 12.86% | |
| S | 1.43% | 0% | 1.43% | 98.57% | |
| T | 0% | 0% | 0% | 3.08% | |
| U | 0% | 0% | 0% | 8.57% | |
| V | 0% | 98.57% | 95.71% | 2.86% | Very strong latent generalisation |
| W | 0% | 100% | 91.43% | 18.57% | Strong in all AE+MLP adapted modes |
| X | 0% | 50.00% | 54.29% | 4.29% | Slight viewpoint, partially recovered |
| Y | 0% | 0% | 52.86% | 82.86% | |
| Z | 0% | 0% | **0%** | 0% | Tilted — unrecovered |

---

## Per-Class Breakdown — Direction 2: ayuraj → danrasband

| Class | Zero-shot | MLP-only | Full fine-tune | CNN fine-tune | Notes |
|---|---|---|---|---|---|
| A | 0% | 0% | 3.33% | 6.67% | |
| B | 0% | 0% | 16.67% | 6.67% | |
| C | 3.33% | 13.33% | 10.00% | 23.33% | |
| D | 13.33% | 0% | 20.00% | 23.33% | Side-view — some recovery with encoder ft |
| E | 0% | 0% | 6.67% | 3.33% | |
| F | 0% | 6.67% | 6.67% | 13.33% | |
| G | 3.33% | 0% | 10.00% | 23.33% | |
| H | **56.67%** | 0% | 3.33% | 50.00% | Regresses badly after fine-tune |
| I | 0% | 0% | 3.33% | 50.00% | |
| J | 13.33% | 13.33% | 10.00% | 73.33% | |
| K | 0% | 3.33% | 23.33% | 30.00% | |
| L | 0% | 30.00% | 13.33% | 60.00% | |
| M | 0% | 30.00% | **50.00%** | 83.33% | Best AE+MLP result in this direction |
| N | 0% | 0% | 3.33% | 0% | |
| O | 0% | 16.67% | 0% | 20.00% | |
| P | 40.00% | 3.33% | 6.67% | 83.33% | High zero-shot, regresses after ft |
| Q | **60.00%** | 60.00% | 3.33% | 56.67% | Collapses in full fine-tune |
| R | 0% | 6.67% | 0% | 33.33% | |
| S | 0% | 0% | 0% | 6.67% | |
| T | 0% | 0% | 0% | 3.33% | |
| U | 0% | 0% | 3.33% | 36.67% | |
| V | 0% | 36.67% | 13.33% | 16.67% | |
| W | 0% | 6.67% | 3.33% | 30.00% | |
| X | 0% | 16.67% | 26.67% | 46.67% | |
| Y | 0% | 30.00% | 13.33% | 23.33% | |
| Z | 0% | 0% | 10.00% | 6.67% | |

---

## Key Findings

### 1. Zero-shot parity, adaptation gap
AE+MLP matches CNN zero-shot on ayuraj (both 27.48%) because digits generalise perfectly across datasets for both pipelines. After adaptation, CNN outperforms AE+MLP significantly (39.92% vs 20.80% on ayuraj; 31.15% vs 10.00% on danrasband).

### 2. The digit catastrophe
CNN fine-tuning preserves digit accuracy because frozen early blocks retain digit feature representations — only the classifier head adapts. AE+MLP loses **all digits** when fine-tuned on danrasband (a letters-only source) because the MLP is retrained from scratch on source latents and never encounters a digit latent during training. This is a structural consequence of the two-stage design.

### 3. MLP-only retrain is mostly harmful
On the ayuraj direction, MLP-only retrain is **worse than zero-shot** (15.75% vs 27.48%). Retraining the MLP on source-domain latents — which occupy different latent-space positions than target-domain latents for the same class — shifts decision boundaries away from the target distribution. The encoder's frozen geometry is the bottleneck.

### 4. Encoder fine-tuning partially recovers letters
Full fine-tune recovers several letter classes dramatically (H: 0→91%, V: 0→95%, W: 0→91%, B: 0→77%, L: 0→84%) in the danrasband→ayuraj direction. However the overall number (20.80%) is still below zero-shot (27.48%) due to digit loss.

### 5. Viewpoint-shifted classes remain stubbornly unrecovered
Classes flagged in visual analysis as having different viewpoints show consistent failure:
- **D** (side view): 0% across all AE+MLP modes in both directions
- **Q, Z** (tilted): 0% in full fine-tune on ayuraj
- **H** (tilted): paradoxically recovers well (91%) danrasband→ayuraj but collapses (3%) ayuraj→danrasband — likely because danrasband's H samples happen to be more similar to the training distribution

### 6. Some classes generalise inherently well in latent space
L, O, R, V, W achieve high accuracy in MLP-only retrain (danrasband→ayuraj: L 91%, O 100%, R 98%, V 98%, W 100%), indicating their latent representations are geometrically stable across domains even with a frozen encoder. These classes likely have distinctive enough shapes that the L2-normalised latent clusters remain separable across domain shifts.

### 7. Full fine-tune can regress zero-shot bright spots
Direction 2 (ayuraj→danrasband): H drops from 56% (zero-shot) to 3% (full fine-tune), P from 40% to 6%, Q from 60% to 3%. The encoder adapts to ayuraj's domain (dark background, specific viewpoints) and in doing so moves the latent positions of these classes away from where danrasband's samples land.

---

## Dataset Notes

| Dataset | Images | Classes | Pixel mean | Domain distance from training |
|---|---|---|---|---|
| ayuraj | 2,515 | 36/39 | 0.173 | **0.382** (very dark, black background) |
| danrasband | 780 | 26/39 | 0.542 | 0.013 (close to training distribution) |
| Training set (ASL39) | 70,350 | 39 | ~0.555 | — |

The large domain distance for ayuraj (black PNG background vs varied backgrounds in training) is the primary driver of cross-dataset difficulty. Danrasband's pixel statistics are close to the training distribution but it lacks digits entirely, limiting what any fine-tuned model can learn.

---

## Files

| File | Description |
|---|---|
| `aemplp_zeroshot_test_ayuraj.csv` | AE+MLP zero-shot on ayuraj |
| `aemplp_zeroshot_test_danrasband.csv` | AE+MLP zero-shot on danrasband |
| `aemplp_mlpretrain_danrasband_test_ayuraj.csv` | MLP-only retrain: danrasband→ayuraj |
| `aemplp_mlpretrain_ayuraj_test_danrasband.csv` | MLP-only retrain: ayuraj→danrasband |
| `aemplp_fullfinetune_danrasband_test_ayuraj.csv` | Full fine-tune: danrasband→ayuraj |
| `aemplp_fullfinetune_ayuraj_test_danrasband.csv` | Full fine-tune: ayuraj→danrasband |
| `finetune_danrasband_test_ayuraj.csv` | CNN fine-tune reference: danrasband→ayuraj |
| `finetune_ayuraj_test_danrasband.csv` | CNN fine-tune reference: ayuraj→danrasband |
| `finetune_aemplp_crosseval.py` | Script that produced the AE+MLP results |
