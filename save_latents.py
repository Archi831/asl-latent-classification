from pathlib import Path

import numpy as np
import torch

from data_prep import load_preprocessed_datasets
from autoencoder import build_autoencoder

DATA_DIR   = r"D:\ns\asl_alphabet_train"
IMG_SIZE   = (64, 64)
BATCH_SIZE = 128
SEED       = 67
LATENT_DIM = 128

MODELS_DIR  = Path("models")
LATENTS_DIR = Path("latents")
LATENTS_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT = MODELS_DIR / "best_autoencoder.pth"

def extract_latents(encoder, dataloader, device):
    encoder.eval()
    all_z      = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            z      = encoder(images)
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
    encoder     = autoencoder.encoder

    print(f"Loaded checkpoint: {CHECKPOINT}")

    encoder_path = MODELS_DIR / "encoder_only1.pth"
    torch.save(encoder.state_dict(), encoder_path)
    print(f"Encoder weights saved → {encoder_path}")

    print("\nExtracting latents …")

    splits = {
        "train": train_loader,
        "val":   val_loader,
        "test":  test_loader,
    }

    all_z_parts, all_label_parts = [], []

    for split_name, loader in splits.items():
        z, labels = extract_latents(encoder, loader, device)
        all_z_parts.append(z)
        all_label_parts.append(labels)
        print(f"  {split_name:5s}: {z.shape[0]:>6,} samples  →  z shape {z.shape}")

    latent_vectors = np.concatenate(all_z_parts,     axis=0)
    latent_labels  = np.concatenate(all_label_parts, axis=0)
    class_names_np = np.array(class_names)

    vec_path    = LATENTS_DIR / "latent_vectors.npy"
    labels_path = LATENTS_DIR / "latent_labels.npy"
    names_path  = LATENTS_DIR / "latent_class_names.npy"

    np.save(vec_path,    latent_vectors)
    np.save(labels_path, latent_labels)
    np.save(names_path,  class_names_np)

    print(f"\nSaved:")
    print(f"  {vec_path}    shape: {latent_vectors.shape}")
    print(f"  {labels_path} shape: {latent_labels.shape}")
    print(f"  {names_path}  shape: {class_names_np.shape}")
    print("\nDone.")

    loaded = np.load(vec_path)
    assert loaded.shape == latent_vectors.shape, "Sanity check failed!"
    print("Sanity check passed ✓")