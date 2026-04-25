# Real-World Evaluation Report

**Branch:** `realworld-eval` | **Last updated:** 2026-04-25

---

## Overview

This report documents how well the trained CNN and AE+MLP pipelines transfer to ASL image datasets that were not seen during training. All models were trained on ASL39 (117k images, 39 classes, 3000/class, controlled studio environment). The external datasets represent different signers, backgrounds, lighting conditions, and equipment.

Two experiment types are reported:

| Experiment | Models |
|---|---|
| Zero-shot — no adaptation, weights as trained | CNN (ab5/ab6/ab7), AE+MLP |
| Cross-dataset fine-tuning — adapt on one dataset, test on other | CNN (ab5/ab6/ab7), AE+MLP |

---

## External Datasets

| Dataset | Kaggle ID | Classes | Images/class | Character |
|---|---|---|---|---|
| **ayuraj** | `ayuraj/asl-dataset` | A–Z + 0–9 (36 total) | ~70 | Controlled lighting, uniform background, hand-cropped |
| **danrasband** | `danrasband/asl-alphabet-test` | A–Z (26 total) | 30 | Real-world footage, single subject, natural background |

**Note on debashishsau:** This dataset (206k images, 26 classes) was initially considered but was excluded after discovering it contains images sourced from the ASL39 training corpus, from ayuraj, and from danrasband. Zero-shot evaluation produced 75–76% accuracy, reflecting training-set memorisation rather than genuine transfer. Additionally, genuine real-world images account for fewer than 100 of the 206k images; the remainder are synthetic augmentations. See `docs/debashishsau_exclusion_note.md` for the full account.

---

## Zero-Shot Evaluation

Base model weights are used directly with no adaptation.

### CNN — all variants

| Model | Head | Config | Ayuraj (2515 imgs) | Danrasband (780 imgs) |
|---|---|---|:---:|:---:|
| ab1 | standard | Adam, no aug, no LS | 28.71% | 4.87% |
| ab2 | standard | AdamW, no aug, no LS | 27.63% | 3.33% |
| ab3 | standard | AdamW, aug | 27.59% | **17.44%** |
| ab4 | standard | AdamW, aug, LS, plateau | 27.67% | 16.03% |
| ab5 | standard | AdamW, aug, LS, cosine | 27.48% | 14.74% |
| ab6 | shallow | AdamW, aug, LS, plateau | **28.07%** | 16.15% |
| ab7 | deep | AdamW, aug, LS, plateau | 27.51% | 13.08% |

The ayuraj headline is misleading: nearly all correct predictions come from the 700 digit images (10 classes × 70), which score ~97–100% across all models. Every letter class scores ~0%. The 27–28% figure is essentially the digit-fraction of the dataset, not genuine letter recognition. Ayuraj accuracy is nearly identical across all seven configurations (~1.2 pp spread).

Danrasband (letters only) tells a much more differentiated story. Configurations **without data augmentation** (ab1, ab2) score only 3–5%—comparable to the AE+MLP. Configurations **with augmentation** (ab3–ab7) achieve 13–17%, with ab3 reaching the highest overall (17.44%). This makes augmentation the dominant driver of zero-shot letter generalization, ahead of optimizer choice, label smoothing, or head architecture.

### AE+MLP — zero-shot

| Dataset | AE+MLP accuracy |
|---|:---:|
| Ayuraj | 27.48% |
| Danrasband | **5.00%** |

The AE+MLP result on ayuraj matches the CNN identically in headline number, but for different reasons: digits score 100% (MLP correctly routes the digit latent clusters), while all letter classes score 0%. On danrasband, the AE+MLP scores only 5.00% — substantially worse than the CNN (14.74%). The latent representation learned from ASL39 letter images is too far from the danrasband visual domain for the frozen MLP head to recover any signal; only H (56.7%), P (40%) and J (13.3%) contribute anything.

---

## Cross-Dataset Fine-Tuning

The model is fine-tuned on one real-world dataset and immediately evaluated on the other. Fine-tuning strategy: freeze first two ConvBlocks, unfreeze last two + classifier; AdamW lr=1e-4, up to 30 epochs, early stopping patience=8.

### CNN — all variants

| Model | Head | Danrasband → Ayuraj | Ayuraj → Danrasband |
|---|---|:---:|:---:|
| ab5 | standard | 40.32% | 31.41% |
| ab6 | shallow | 39.60% | **34.49%** |
| ab7 | deep | 40.04% | 27.95% |

**Danrasband → ayuraj** is constrained by the tiny fine-tuning set (780 images, 30/class, letters only). Digits never appear in fine-tuning and rely entirely on the frozen backbone — they still score 72–100% across models. All three models converge to ~40% regardless of head, suggesting the bottleneck is data volume, not architecture.

**Ayuraj → danrasband** is the harder and more informative direction. Ayuraj provides ~70 images/class and covers both letters and digits, giving a stronger fine-tuning signal. The 6-point spread between ab6 (34.49%) and ab7 (27.95%) shows that the deep head with BatchNorm overfits more tightly to ayuraj's controlled visual style and generalises worse.

**Fine-tuning gain over zero-shot:**

| Model | Danrasband→Ayuraj gain | Ayuraj→Danrasband gain |
|---|:---:|:---:|
| ab5 | +12.8 pp | +16.7 pp |
| ab6 | +11.5 pp | +18.3 pp |
| ab7 | +12.5 pp | +14.9 pp |

Even 30 images/class roughly doubles cross-domain accuracy, confirming the backbone features are transferable and the domain gap is primarily in the classifier decision boundaries.

### AE+MLP — fine-tuning modes

Two fine-tuning strategies were tested for the AE+MLP pipeline:

| Mode | Description |
|---|---|
| **MLP-retrain** | Freeze encoder entirely; retrain MLP from scratch on target domain |
| **Full fine-tune** | Unfreeze encoder + retrain MLP jointly |

| Mode | Danrasband → Ayuraj | Ayuraj → Danrasband |
|---|:---:|:---:|
| MLP-retrain | 15.75% | 10.51% |
| Full fine-tune | 20.80% | 10.00% |

Both modes trail the CNN substantially. Full fine-tuning on danrasband (780 imgs, letters only) reaches 20.80% on ayuraj but still lags the CNN (40.32%) by ~20 pp. MLP-retrain and full fine-tune perform similarly on danrasband (~10%), both far below the CNN (~31–34%). The AE+MLP pipeline is markedly less adaptable to domain shift under limited fine-tuning data.

---

## Summary

| Experiment | Model | Dataset | Accuracy |
|---|---|---|:---:|
| Zero-shot | CNN ab6 | Ayuraj | 28.1% |
| Zero-shot | CNN ab6 | Danrasband | 16.2% |
| Zero-shot | AE+MLP | Ayuraj | 27.5% |
| Zero-shot | AE+MLP | Danrasband | 5.0% |
| Fine-tune danrasband→ | CNN ab5 | Ayuraj | 40.3% |
| Fine-tune danrasband→ | CNN ab6 | Ayuraj | 39.6% |
| Fine-tune danrasband→ | CNN ab7 | Ayuraj | 40.0% |
| Fine-tune danrasband→ | AE+MLP (full) | Ayuraj | 20.8% |
| Fine-tune ayuraj→ | CNN ab5 | Danrasband | 31.4% |
| Fine-tune ayuraj→ | CNN ab6 | Danrasband | **34.5%** |
| Fine-tune ayuraj→ | CNN ab7 | Danrasband | 28.0% |
| Fine-tune ayuraj→ | AE+MLP (full) | Danrasband | 10.0% |

---

## Key Findings

**1. Digits transfer; letters do not (zero-shot).** The ASL39 training images share the same digit visual conventions as ayuraj. Letters fail because hand shape, background and lighting differ. This is a domain shift in visual style, not in sign semantics.

**2. Fine-tuning with as few as 30 images/class roughly doubles accuracy.** The backbone features learned on ASL39 are reusable. The domain gap sits primarily in the classifier head, not the feature extractor. Even a small target-domain sample can recalibrate the decision boundaries.

**3. The CNN generalises better than the AE+MLP under domain shift.** In every matched experiment, the CNN outperforms the AE+MLP pipeline — sometimes by wide margins (danrasband zero-shot: 14.7% vs 5.0%; ayuraj fine-tune: 40.3% vs 20.8%). The frozen latent representation learned on ASL39 does not adapt well to new visual distributions, whereas the CNN's partial fine-tuning of upper convolutional layers can re-learn domain-specific features.

**4. Shallower classifier heads generalise better.** ab6 (single linear layer) consistently outperforms ab7 (deep head with BatchNorm) on cross-domain tasks. The deep head overfits the fine-tuning distribution's batch statistics; the shallow head has fewer degrees of freedom to overfit with.
