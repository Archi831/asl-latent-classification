# Research and Experiment Plan

This document records the dataset investigation, the completed CNN hyperparameter sweep, and the current documentation draft structure. Sections that are not yet finished are intentionally left as placeholders.

## 1. Problem Statement

We study image-based classification of sign language gestures with two model families:

- a direct CNN classifier trained end-to-end on images
- an autoencoder + classifier pipeline that compresses images into a latent representation before classification

The central question is whether a compact learned representation can match or improve the direct CNN baseline, and whether either approach is more robust under perturbations.

## 2. Dataset Research

### 2.1 Current Dataset: ASL Alphabet

The current project dataset is the Kaggle ASL Alphabet set.

- 87,000 training images
- 29 classes: A-Z, Space, Delete, Nothing
- 200x200 source images
- used in the project as 64x64 grayscale inputs

This dataset is strong for a course project, but in this codebase the CNN reaches a perfect clean test score, so clean accuracy is no longer a useful discriminator for research claims.

### 2.2 Candidate Dataset: Sign Language MNIST

The Kaggle Sign Language MNIST dataset is not a strong next benchmark for this project.

- 27,455 training cases and 7,172 test cases
- 24 classes, excluding J and Z because they require motion
- 28x28 grayscale images
- Kaggle notebooks linked on the page already report 97% to 100% accuracy

Current finding after local runs in this repository: the dataset does not always hit 100% test accuracy under all settings, but it can still become near-saturated quickly. It is useful as a sanity-check benchmark, not as the strongest final research benchmark.

### 2.3 Better American Sign Language Alternatives

The following alternatives are more suitable for research:

| Dataset | Why it is better | Size / Classes |
|---|---|---|
| ASL-alphabet-numbers-dataset | Adds digits 0-9, increases the class count, still static American signs | about 101k files, 39 classes, 224x224 grayscale |
| ASL(American Sign Language) Alphabet Dataset by Debashish Sau | Larger and more varied than the current ASL Alphabet benchmark | about 223k files, 29 classes |
| ASL-Alphabet-Dataset by Khansa Tehreem | Higher resolution and static alphabet-only setup | about 71.7k files, 300x300 JPEG, 26 classes |

### 2.4 Recommendation

Recommended next dataset:

- primary choice: ASL-alphabet-numbers-dataset
- fallback: Debashish Sau ASL(American Sign Language) Alphabet Dataset

Reasoning:

- both remain American Sign Language
- both are static-image datasets, so they fit the current CNN/autoencoder pipeline
- both are more research-worthy than Sign Language MNIST
- the 39-class dataset is especially useful if the goal is to avoid clean-accuracy saturation

## 3. Current Hyperparameter Sweep

The following 7 ablations were completed on the current ASL Alphabet benchmark.

| Run | Optimizer | Dropout | Augment | Label Smoothing | Scheduler | Best Val Acc | Test Acc | Macro F1 |
|---|---|---:|---|---:|---|---:|---:|---:|
| ab1_baseline | Adam | 0.5 | No | 0.00 | Plateau | 1.0000 | 1.0000 | 1.0000 |
| ab2_adamw_wd | AdamW | 0.5 | No | 0.00 | Plateau | 1.0000 | 1.0000 | 1.0000 |
| ab3_adamw_wd_do03 | AdamW | 0.3 | No | 0.00 | Plateau | 1.0000 | 1.0000 | 1.0000 |
| ab4_aug | AdamW | 0.3 | Yes | 0.00 | Plateau | 1.0000 | 1.0000 | 1.0000 |
| ab5_aug_ls005 | AdamW | 0.3 | Yes | 0.05 | Plateau | 1.0000 | 1.0000 | 1.0000 |
| ab6_aug_ls005_cosine | AdamW | 0.3 | Yes | 0.05 | Cosine | 1.0000 | 1.0000 | 1.0000 |
| ab7_aug_ls005_bs64 | AdamW | 0.3 | Yes | 0.05 | Plateau | 1.0000 | 1.0000 | 1.0000 |

### Interpretation

The sweep shows that the current ASL Alphabet clean test split is saturated. The main differences between runs are convergence speed and training stability, not final clean accuracy.

Most useful settings so far:

- AdamW with weight decay 1e-4
- dropout 0.3
- light augmentation
- label smoothing 0.05 when the goal is calibration or robustness

## 4. Sign Language MNIST Hyperparameter Tests

Implemented script:

- `cnn_classifier/train_sign_mnist.py`

Completed runs:

| Run | Optimizer | Scheduler | Dropout | Label Smoothing | Batch | Augment | Test Acc | Test Macro F1 |
|---|---|---|---:|---:|---:|---|---:|---:|
| sm1_adam_baseline | Adam | Plateau | 0.5 | 0.00 | 128 | No | 0.9968 | 0.9960 |
| sm2_adamw_wd | AdamW | Plateau | 0.3 | 0.00 | 128 | No | 0.9909 | 0.9911 |
| sm3_sgd_cosine | SGD | Cosine | 0.3 | 0.00 | 128 | No | 0.9868 | 0.9859 |
| sm4_aug_ls_e8 | AdamW | Plateau | 0.3 | 0.05 | 128 | Yes | 1.0000 | 1.0000 |
| sm5_aug_ls_bs64_e8 | AdamW | Plateau | 0.3 | 0.05 | 64 | Yes | 1.0000 | 1.0000 |

Interpretation:

- Your skepticism was correct: unlike the original ASL Alphabet setup, this dataset produced several runs below 100% test accuracy.
- However, with regularization and augmentation, it can still reach perfect test scores in short runs.
- This confirms that Sign Language MNIST is better than a fully saturated benchmark, but still not ideal as the final main dataset for strong research claims.

Remaining TODO for stronger evidence:

- run a full-length augmented sweep (without runtime interruption)
- add confusion matrices for each run
- evaluate robustness with synthetic corruption

## 5. Methodology

### 5.1 Data Preparation

- ASL Alphabet: folder dataset, resized to 64x64 grayscale in existing pipeline
- Sign Language MNIST: CSV dataset with 28x28 grayscale images, values scaled to [0, 1]
- Sign Language MNIST split in current script:
	- official Kaggle test CSV used as test set
	- 90/10 split of training CSV into train/validation with seed 67
- [TODO] final primary dataset decision for the report

### 5.2 Metrics

- Accuracy
- Macro F1
- Confusion matrix
- Validation loss curves
- [TODO] robustness metrics under corruption

## 6. Results

### 6.1 CNN Results

Completed on:

- ASL Alphabet (saturated clean accuracy)
- Sign Language MNIST (mixed results from 98.68% to 100%)

TODO: add consolidated comparison charts across both datasets.

### 6.2 Autoencoder + MLP Results

TODO: run the latent-space pipeline on the chosen dataset and compare against the CNN baseline.

### 6.3 Learning Curves

TODO: insert train/validation curves for both model families.

### 6.4 Error Analysis

TODO: add confusion matrix analysis and describe the most frequently confused classes.

## 7. Critical Analysis and Discussion

TODO: explain why the models succeed or fail on the selected dataset.

Possible discussion points:

- class similarity and hand pose ambiguity
- dataset difficulty and signer variation
- effect of augmentation on robustness
- whether the latent bottleneck improves interpretability or generalization

## 8. Limitations and Future Work

- [TODO] dataset-specific limitations
- [TODO] robustness experiments on corrupted inputs
- [TODO] cross-dataset generalization
- [TODO] larger latent-space ablation, for example 128 vs. 64 vs. 32

## 9. Team Contributions

| Person | Contribution | Status |
|---|---|---|
| Lana | Autoencoder pipeline | Done |
| Archie | CNN classifier and documentation | In progress |
| Lilia | MLP on latent vectors | TODO |
| Maksym | t-SNE and robustness testing | TODO |

## 10. References

TODO: add citations for the final dataset(s), PyTorch, scikit-learn, Kaggle dataset pages, and any papers used in the discussion.

## 11. Generated Artifacts

Existing ASL Alphabet artifacts:

- `cnn_classifier/models/best_cnn_classifier_ab*.pth`
- `cnn_classifier/outputs/plots/cnn_training_curves_ab*.png`
- `cnn_classifier/outputs/ablation_results.csv`

Sign Language MNIST artifacts:

- `data/sign-language-mnist/`
- `cnn_classifier/models/best_sign_mnist_*.pth`
- `cnn_classifier/outputs/sign_mnist_ablation_results.csv`

Additional downloaded dataset:

- `data/asl-alphabet-numbers/asl-numbers-alphabet-dataset/` (39 classes, ready for next-stage experiments)

TODO: add the same artifact structure for the final selected follow-up dataset (likely 39-class ASL alphabet+numbers).
