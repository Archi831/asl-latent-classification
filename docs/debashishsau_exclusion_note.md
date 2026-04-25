# Exclusion of the debashishsau Dataset

## Dataset

`debashishsau/aslamerican-sign-language-aplhabet-dataset` (Kaggle) — 206,137 images across 26 letter classes.

## Why It Was Considered

The dataset is large and covers the full A–Z alphabet, making it an attractive candidate for cross-dataset generalisation experiments alongside ayuraj and danrasband.

## Why It Was Excluded

Three compounding issues made the dataset unsuitable for any role in our evaluation pipeline.

**Training-set overlap.** Inspection revealed that the dataset contains images drawn from the same source as the ASL39 training corpus used to train our models. Evaluating a model on data it has already seen during training inflates accuracy and produces invalid generalisation estimates. Zero-shot evaluation yielded 75–76% accuracy — a number that reflects memorisation rather than transfer, consistent with training-set leakage.

**Cross-dataset overlap.** The dataset also contains images sourced from the danrasband and ayuraj datasets. This makes any cross-evaluation between debashishsau and either of those datasets contaminated in both directions: fine-tuning on debashishsau leaks test-set information, and evaluating on debashishsau after training on ayuraj or danrasband leaks training information.

**Composition imbalance.** The overwhelming majority of images are synthetic: controlled-environment studio shots subjected to programmatic augmentation (cropping, translation, rotation). Genuine real-world images account for fewer than 100 samples in the entire dataset. This extreme imbalance between synthetic and real-world content makes the dataset unsuitable as a proxy for real-world robustness evaluation, which is the primary goal of our external test set experiments.

## Conclusion

debashishsau was dropped from all experiments. The valid external evaluation datasets in this work are **ayuraj** (controlled lighting, uniform background, ~70 images per class, 36 classes) and **danrasband** (real-world footage, single subject, ~30 images per class, 26 letter classes only).
