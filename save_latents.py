from pathlib import Path

import numpy as np
import torch

from data_prep import load_preprocessed_datasets
from autoencoder import build_autoencoder

DATA_DIR   = r"D:\ns\asl_alphabet_train"
IMG_SIZE   = (64,64)
BATCH_SIZE = 128
SEED       = 67
LATENT_DIM = 128

MODELS_DIR  = Path("models")
LATENTS_DIR = Path("latents")
LATENTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = MODELS_DIR / "best_autoencoder.pth"

def extract_latents(encoder, dataloader, device):
    encoder.eval()
    all_z = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            z = encoder(images)
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_z, axis=0), np.concatenate(all_labels, axis=0)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = load_preprocessed_datasets(
        DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED
    )

    encoder, _, autoencoder = build_autoencoder(latent_dim=LATENT_DIM)
    autoencoder.load_state_dict(torch.load(CHECKPOINT, map_location=device))
    autoencoder = autoencoder.to(device)
    encoder = autoencoder.encoder

    print(f"Loaded checkpoint: {CHECKPOINT}")

    encoder_path = MODELS_DIR / "encoder_only1.pth"
    torch.save(encoder.state_dict(), encoder_path)
    print(f"Encoder weights saved → {encoder_path}")

    print("\nExtracting latents...")

    splits = {
        "train": train_loader,
        "val": test_loader,
        "test": test_loader,
    }

    for split_name, loader in splits.items():
        z, labels = extract_latents(encoder, loader, device)

        vec_path = LATENTS_DIR / f"{split_name}_latent_vectors.npy"
        labels_path = LATENTS_DIR / f"{split_name}_latent_labels.npy"

        np.save(vec_path, z)
        np.save(labels_path, labels)

        print(f"{split_name.capitalize()} saved:")
        print(f"  {vec_path}    shape: {z.shape}")
        print(f"  {labels_path} shape: {labels.shape}")

        loaded_z = np.load(vec_path)
        assert loaded_z.shape == z.shape, f"Sanity check failed for {split_name} vectors!"

    names_path = LATENTS_DIR / "latent_class_names.npy"
    np.save(names_path, np.array(class_names))
    print(f"\nClass names saved:")
    print(f"  {names_path}  shape: {np.array(class_names).shape}")

    print("\nAll splits saved separately ✓")