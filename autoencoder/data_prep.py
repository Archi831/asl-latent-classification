# Import necessary libraries for file paths, random operations, numerical operations,
# deep learning, dataset handling, and visualization

from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DATA_DIR = r"D:\ns\asl_alphabet_train"
IMG_SIZE = (64, 64)
BATCH_SIZE = 128
SEED = 67

# Set random seeds across all libraries to ensure reproducible results
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Define data preprocessing pipeline (transformations)
def get_transforms(img_size=(64, 64), augment=False):
    base = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
    ]
    if augment:
        base.extend([
            transforms.RandomAffine(degrees=8, translate=(0.08, 0.08), scale=(0.92, 1.08), shear=5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
        ])
    base.append(transforms.ToTensor())
    if augment:
        base.append(transforms.RandomErasing(p=0.1, scale=(0.02, 0.08), value=0))
    return transforms.Compose(base)


# Load dataset, split into train/val/test sets, and create data loaders
def load_preprocessed_datasets(
    data_dir,
    img_size=(64, 64),
    batch_size=128,
    seed=67,
    augment=False,
    num_workers=0,
    expected_num_classes=None,
):
    set_seed(seed)
    data_dir = Path(data_dir)

    # Determine indices once from an untransformed dataset, then attach per-split transforms
    base_dataset = datasets.ImageFolder(root=data_dir, transform=None)
    class_names  = base_dataset.classes
    num_classes  = len(class_names)

    print(f"\nFound {num_classes} classes:")
    print(class_names)

    if expected_num_classes is not None and num_classes != expected_num_classes:
        print(f"WARNING: Expected {expected_num_classes} classes, but found {num_classes}.")

    total_size = len(base_dataset)
    train_size = int(0.70 * total_size)
    val_size   = int(0.10 * total_size)
    test_size  = total_size - train_size - val_size

    generator  = torch.Generator().manual_seed(seed)
    all_indices = torch.randperm(total_size, generator=generator).tolist()
    train_idx   = all_indices[:train_size]
    val_idx     = all_indices[train_size:train_size + val_size]
    test_idx    = all_indices[train_size + val_size:]

    train_tf = get_transforms(img_size, augment=augment)
    eval_tf  = get_transforms(img_size, augment=False)

    full_train = datasets.ImageFolder(root=data_dir, transform=train_tf)
    full_eval  = datasets.ImageFolder(root=data_dir, transform=eval_tf)

    train_dataset = torch.utils.data.Subset(full_train, train_idx)
    val_dataset   = torch.utils.data.Subset(full_eval,  val_idx)
    test_dataset  = torch.utils.data.Subset(full_eval,  test_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_names

# Get the number of samples in a dataloader
def count_samples(dataloader):
    return len(dataloader.dataset)

# Visualize a grid of sample images with their labels
def show_sample_images(dataloader, class_names, num_images=9):
    images, labels = next(iter(dataloader))

    plt.figure(figsize=(8, 8))
    for i in range(min(num_images, len(images))):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].squeeze(0), cmap="gray")
        plt.title(class_names[labels[i].item()])
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    # Load the dataset and create data loaders

    train_loader, val_loader, test_loader, class_names = load_preprocessed_datasets(
        DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    # Count and display the number of samples in each dataset split

    train_count = count_samples(train_loader)
    val_count = count_samples(val_loader)
    test_count = count_samples(test_loader)
    total_count = train_count + val_count + test_count

    print("\nDataset sizes:")
    print(f"Train: {train_count}")
    print(f"Validation: {val_count}")
    print(f"Test: {test_count}")
    print(f"Total: {total_count}")

    images, labels = next(iter(train_loader))
    print("\nOne training batch:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Min pixel value:", images.min().item())
    print("Max pixel value:", images.max().item())

    show_sample_images(train_loader, class_names)