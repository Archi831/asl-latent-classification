from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

DATA_DIR = str(Path(__file__).resolve().parent.parent / "data" / "asl-alphabet-numbers" / "asl-numbers-alphabet-dataset")
IMG_SIZE = (64, 64)
BATCH_SIZE = 128
SEED = 67
NOISE_FACTOR = 0.25  # <-- control noise strength here


# Set random seeds
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Add Gaussian noise
def add_noise(image, noise_factor=NOISE_FACTOR):
    noise = noise_factor * torch.randn_like(image)
    noisy_image = image + noise
    noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
    return noisy_image


# Transform pipeline WITH noise inside
def get_transforms(img_size=(64, 64)):
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: add_noise(x)),  # <-- THIS is the key line
    ])


# Load dataset
def load_preprocessed_datasets(data_dir, img_size=(64, 64), batch_size=128, seed=67):
    set_seed(seed)

    data_dir = Path(data_dir)
    transform = get_transforms(img_size)

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    class_names = full_dataset.classes
    num_classes = len(class_names)

    print(f"\nFound {num_classes} classes:")
    print(class_names)

    if num_classes not in [36, 39]:
        print(f"WARNING: Expected 36 or 39 classes, but found {num_classes}.")

    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.10 * total_size)
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, class_names


def count_samples(dataloader):
    return len(dataloader.dataset)


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

    train_loader, val_loader, test_loader, class_names = load_preprocessed_datasets(
        DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    train_count = count_samples(train_loader)
    val_count = count_samples(val_loader)
    test_count = count_samples(test_loader)

    print("\nDataset sizes:")
    print(f"Train: {train_count}")
    print(f"Validation: {val_count}")
    print(f"Test: {test_count}")
    print(f"Total: {train_count + val_count + test_count}")

    images, labels = next(iter(train_loader))
    print("\nOne training batch:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Min pixel value:", images.min().item())
    print("Max pixel value:", images.max().item())

    show_sample_images(train_loader, class_names)