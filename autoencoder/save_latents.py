from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import torch

from data_prep import load_preprocessed_datasets
from autoencoder import build_autoencoder

_REPO_ROOT  = Path(__file__).resolve().parent.parent
DATA_DIR    = str(_REPO_ROOT / "data" / "asl-alphabet-numbers" / "asl-numbers-alphabet-dataset")
IMG_SIZE    = (64, 64)
BATCH_SIZE  = 128
SEED        = 67
LATENT_DIM  = 128
NUM_CLASSES = 39

MODELS_DIR  = Path(__file__).parent / "models"
LATENTS_DIR = _REPO_ROOT / "latents"


def extract_latents(encoder, dataloader, device):
    encoder.eval()
    all_z, all_labels = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            z      = encoder(images)
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_z, axis=0), np.concatenate(all_labels, axis=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--latent_dim", type=int, default=LATENT_DIM)
    parser.add_argument("--run_name",   type=str, default=None,
                        help="Run name used when saving the checkpoint, e.g. ae_asl39_ld128")
    args = parser.parse_args()

    latent_dim = args.latent_dim
    run_name   = args.run_name or f"ae_asl39_ld{latent_dim}"
    checkpoint = MODELS_DIR / f"best_autoencoder_{run_name}.pth"

    LATENTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader, test_loader, class_names = load_preprocessed_datasets(
        DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED
    )

    _, _, autoencoder = build_autoencoder(latent_dim=latent_dim, num_classes=NUM_CLASSES)
    autoencoder.load_state_dict(torch.load(checkpoint, map_location=device))
    autoencoder = autoencoder.to(device)
    encoder     = autoencoder.encoder

    print(f"Loaded checkpoint: {checkpoint}")

    encoder_path = MODELS_DIR / f"encoder_{run_name}.pth"
    torch.save(encoder.state_dict(), encoder_path)
    print(f"Encoder weights saved -> {encoder_path}")

    print("\nExtracting latents ...")
    splits = {"train": train_loader, "val": val_loader, "test": test_loader}
    all_z_parts, all_label_parts = [], []

    for split_name, loader in splits.items():
        z, labels = extract_latents(encoder, loader, device)
        all_z_parts.append(z)
        all_label_parts.append(labels)
        print(f"  {split_name:5s}: {z.shape[0]:>6,} samples  z shape {z.shape}")

    latent_vectors = np.concatenate(all_z_parts,     axis=0)
    latent_labels  = np.concatenate(all_label_parts, axis=0)
    class_names_np = np.array(class_names)

    vec_path    = LATENTS_DIR / f"latents_asl39_ld{latent_dim}.npy"
    labels_path = LATENTS_DIR / f"labels_asl39_ld{latent_dim}.npy"
    names_path  = LATENTS_DIR / f"class_names_asl39.npy"

    np.save(vec_path,    latent_vectors)
    np.save(labels_path, latent_labels)
    np.save(names_path,  class_names_np)

    print(f"\nSaved:")
    print(f"  {vec_path}    shape: {latent_vectors.shape}")
    print(f"  {labels_path} shape: {latent_labels.shape}")
    print(f"  {names_path}  shape: {class_names_np.shape}")

    loaded = np.load(vec_path)
    assert loaded.shape == latent_vectors.shape, "Sanity check failed!"
    print("Sanity check passed")
