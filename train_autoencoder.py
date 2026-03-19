from pathlib import Path
import copy

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_prep import load_preprocessed_datasets
from autoencoder import build_autoencoder


DATA_DIR = r"D:\ns\asl_alphabet_train"
IMG_SIZE = (64, 64)
BATCH_SIZE = 128
SEED = 67
LATENT_DIM = 128
EPOCHS = 50
LEARNING_RATE = 1e-3

MODELS_DIR = Path("models")
PLOTS_DIR = Path("outputs/plots")

MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def plot_history(history, save_path="outputs/plots/ae_loss_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Autoencoder Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def show_reconstructions(model, dataloader, device, class_names,
                         save_path="outputs/plots/ae_reconstructions.png",
                         num_images=6,
                         skip_nothing=True):
    model.eval()

    nothing_idx = None
    if skip_nothing and "nothing" in [c.lower() for c in class_names]:
        nothing_idx = [c.lower() for c in class_names].index("nothing")

    selected_images = []
    selected_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            for img, lbl in zip(images, labels):
                if nothing_idx is not None and lbl.item() == nothing_idx:
                    continue

                selected_images.append(img)
                selected_labels.append(lbl.item())

                if len(selected_images) == num_images:
                    break

            if len(selected_images) == num_images:
                break

        if len(selected_images) == 0:
            print("No suitable images found for reconstruction preview.")
            return

        images = torch.stack(selected_images).to(device)
        reconstructed = model(images)

    images = images.cpu()
    reconstructed = reconstructed.cpu()

    plt.figure(figsize=(12, 4))
    for i in range(len(images)):
        plt.subplot(2, len(images), i + 1)
        plt.imshow(images[i].squeeze(0), cmap="gray")
        plt.title(f"Original\n{class_names[selected_labels[i]]}")
        plt.axis("off")

        plt.subplot(2, len(images), i + 1 + len(images))
        plt.imshow(reconstructed[i].squeeze(0), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    loop = tqdm(dataloader, desc="Training", leave=False)

    for images, _ in loop:
        images = images.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)

        for images, _ in loop:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, images)

            running_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, images)
            running_loss += loss.item() * images.size(0)

    return running_loss / len(dataloader.dataset)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, test_loader, class_names = load_preprocessed_datasets(
        DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        seed=SEED
    )

    encoder, decoder, autoencoder = build_autoencoder(latent_dim=LATENT_DIM)
    autoencoder = autoencoder.to(device)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
        min_lr=1e-6
    )

    history = {
        "train_loss": [],
        "val_loss": []
    }

    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(autoencoder.state_dict())

    early_stopping_patience = 5
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        train_loss = train_one_epoch(autoencoder, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(autoencoder, val_loader, criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")
        print(f"LR:         {current_lr:.8f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(autoencoder.state_dict())
            torch.save(autoencoder.state_dict(), MODELS_DIR / "best_autoencoder.pth")
            epochs_without_improvement = 0
            print("Best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    autoencoder.load_state_dict(best_model_weights)

    torch.save(autoencoder.state_dict(), MODELS_DIR / "final_autoencoder.pth")
    torch.save(autoencoder.encoder.state_dict(), MODELS_DIR / "encoder_only.pth")
    torch.save(autoencoder.decoder.state_dict(), MODELS_DIR / "decoder_only.pth")

    test_loss = evaluate(autoencoder, test_loader, criterion, device)
    print(f"\nTest MSE Loss: {test_loss:.6f}")

    plot_history(history, save_path=str(PLOTS_DIR / "ae_loss_curve.png"))
    show_reconstructions(
        autoencoder,
        test_loader,
        device,
        class_names=class_names,
        save_path=str(PLOTS_DIR / "ae_reconstructions.png"),
        num_images=6,
        skip_nothing=True
    )