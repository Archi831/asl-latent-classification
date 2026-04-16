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
def get_transforms(img_size=(64, 64)):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),   # Convert RGB to grayscale (ASL signs don't need color)
        transforms.Resize(img_size),                   # Resize all images to uniform dimensions
        transforms.ToTensor(),                         # Convert image to PyTorch tensor and scale to [0, 1]
    ])

# Load dataset, split into train/val/test sets, and create data loaders
def load_preprocessed_datasets(data_dir, img_size=(64, 64), batch_size=128, seed=67):
    set_seed(seed)

    data_dir = Path(data_dir)

    transform = get_transforms(img_size)

    # This automatically assigns labels based on subfolder names
    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Extract class names (e.g., 'A', 'B', 'C', ... 'space', 'del')
    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"\nFound {num_classes} classes:")
    print(class_names)

    if num_classes != 29:
        print(f"WARNING: Expected 29 classes, but found {num_classes}.")

    # Calculate dataset split sizes: 70% training, 10% validation, 20% test
    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    # Randomly split the dataset into training, validation, and test sets
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    # Create DataLoader for training set
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Create DataLoader for validation set
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # Create DataLoader for testing set
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
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