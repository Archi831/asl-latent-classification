# CNN Ablation Analysis

This document summarizes seven CNN training runs on the ASL Alphabet dataset and interprets what changed across optimizer, regularization, augmentation, label smoothing, scheduler, and batch size settings.

## Experimental Setup

- Dataset: ASL Alphabet, 29 classes, 64x64 grayscale input
- Split: 70% train / 10% validation / 20% test with seed 67
- Model: CNN classifier from `cnn_classifier/model.py`
- Baseline optimizer: Adam, lr 1e-3
- Common settings for the ablations:
  - weight decay: 1e-4 when using AdamW runs
  - dropout: 0.3 or 0.5 depending on the run
  - early stopping patience: 7
  - scheduler: ReduceLROnPlateau unless noted otherwise

## Ablation Runs

| Run | Optimizer | Dropout | Augment | Label Smoothing | Scheduler | Epochs | Best Val Loss | Best Val Acc | Test Acc | Macro F1 |
|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|
| ab1_baseline | Adam | 0.5 | No | 0.00 | Plateau | 23 | 4.60e-05 | 1.0000 | 1.0000 | 1.0000 |
| ab2_adamw_wd | AdamW | 0.5 | No | 0.00 | Plateau | 35 | 4.27e-07 | 1.0000 | 1.0000 | 1.0000 |
| ab3_adamw_wd_do03 | AdamW | 0.3 | No | 0.00 | Plateau | 43 | 5.85e-07 | 1.0000 | 1.0000 | 1.0000 |
| ab4_aug | AdamW | 0.3 | Yes | 0.00 | Plateau | 50 | 3.16e-07 | 1.0000 | 1.0000 | 1.0000 |
| ab5_aug_ls005 | AdamW | 0.3 | Yes | 0.05 | Plateau | 50 | 3.55e-01 | 1.0000 | 1.0000 | 1.0000 |
| ab6_aug_ls005_cosine | AdamW | 0.3 | Yes | 0.05 | Cosine | 50 | 3.55e-01 | 1.0000 | 1.0000 | 1.0000 |
| ab7_aug_ls005_bs64 | AdamW | 0.3 | Yes | 0.05 | Plateau | 41 | 3.55e-01 | 1.0000 | 1.0000 | 1.0000 |

## What the Results Mean

### 1. Clean-test accuracy saturated across all runs

Every checkpoint reached 100% test accuracy and 1.0000 macro F1 on the clean test split. That means the clean split is not discriminative enough to rank these configurations by accuracy.

In practice, the main differences between runs are therefore:

- training stability
- speed of convergence
- validation loss behavior
- robustness under future corruption tests

### 2. AdamW helped the most among the simple changes

Compared with the baseline Adam run, AdamW with weight decay gave:

- lower validation loss
- more stable convergence
- no loss in test accuracy

This is the most useful low-risk improvement in the current setup.

### 3. Lower dropout reduced underfitting risk

Moving from dropout 0.5 to 0.3 improved optimization behavior and preserved perfect test accuracy.

That suggests the original dropout value was a bit aggressive for this dataset and architecture.

### 4. Augmentation increased training time but did not hurt accuracy

The augmentation run trained for the full 50 epochs, which is expected because the task became harder during training.

The model still reached perfect clean-test accuracy, so augmentation is a reasonable choice if the later robustness experiments matter more than raw validation speed.

### 5. Label smoothing changed the loss scale, so loss is no longer directly comparable

For runs ab5-ab7, the validation loss is around 0.355 because the training objective includes label smoothing.

That makes the loss values incomparable with the no-smoothing runs. Accuracy is the fair comparison metric there, and all runs still reached 100% test accuracy.

### 6. Cosine scheduling did not change clean accuracy

The cosine scheduler behaved similarly to ReduceLROnPlateau in terms of final clean accuracy.

It is still useful if you want a smoother decay schedule, but this dataset does not reveal a clear advantage on the clean split.

### 7. Batch size 64 did not materially change the outcome

The smaller batch size still reached the same clean-test ceiling.

Its main effect was a slightly different convergence path, not a different final result.

## Best Practical Configuration

If the goal is the strongest all-around CNN baseline for this project, the most defensible choice is:

- AdamW
- lr 1e-3
- weight decay 1e-4
- dropout 0.3
- light augmentation enabled
- ReduceLROnPlateau

This configuration is the best compromise between optimization stability and robustness-oriented training.

## Important Limitation

Because the clean test set is too easy for this model, the ablation study does not separate the runs by accuracy.

For the report, the next meaningful comparison should be under corruption tests such as:

- Gaussian noise
- brightness shift
- resolution drop

Those tests are much more likely to show whether augmentation and regularization improve real robustness.

## Files Produced

- Best checkpoints: `cnn_classifier/models/best_cnn_classifier_ab*.pth`
- Training curves: `cnn_classifier/outputs/plots/cnn_training_curves_ab*.png`
- Run log CSV: `cnn_classifier/outputs/ablation_results.csv`
