from pathlib import Path
import copy # For making a deep copy of model weights
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchmetrics import StructuralSimilarityIndexMeasure

from data_prep import load_preprocessed_datasets
from autoencoder import build_autoencoder

DATA_DIR      = r"D:\ns\asl_alphabet_train"
IMG_SIZE      = (64, 64)
BATCH_SIZE    = 128
SEED          = 67
LATENT_DIM    = 128
EPOCHS        = 50
LEARNING_RATE = 1e-3

SSIM_WEIGHT   = 0.5
MSE_WEIGHT    = 0.5

MODELS_DIR = Path("models")
PLOTS_DIR  = Path("outputs/plots")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.5, ssim_weight: float = 0.5, data_range: float = 1.0):
        super().__init__()
        self.mse_weight  = mse_weight
        self.ssim_weight = ssim_weight
        self.mse         = nn.MSELoss()
        self.ssim        = StructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_loss = self.mse(pred, target)
        ssim_loss = 1.0 - self.ssim(pred, target)
        return self.mse_weight * mse_loss + self.ssim_weight * ssim_loss


def plot_history(history, save_path="outputs/plots/ae_loss_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"],   label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Combined Loss")
    plt.title("Autoencoder Training History")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def show_reconstructions(model, dataloader, device, class_names,
                         save_path="outputs/plots/ae_reconstructions.png",
                         num_images=6, skip_nothing=True):
    model.eval()

    nothing_idx = None
    if skip_nothing and "nothing" in [c.lower() for c in class_names]:
        nothing_idx = [c.lower() for c in class_names].index("nothing")

    selected_images, selected_labels = [], []
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

        if not selected_images:
            print("No suitable images found.")
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
        # Forward pass: reconstruct images
        outputs = model(images)

        # Compute reconstruction loss
        loss = criterion(outputs, images)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        # Accumulate batch loss
        running_loss += loss.item() * images.size(0)

        loop.set_postfix(loss=f"{loss.item():.5f}")

    return running_loss / len(dataloader.dataset)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)
        for images, _ in loop:
            images = images.to(device, non_blocking=True)
            # Forward pass
            outputs = model(images)

            # Compute validation loss
            loss = criterion(outputs, images)

            # Accumulate total loss
            running_loss += loss.item() * images.size(0)

            loop.set_postfix(loss=f"{loss.item():.5f}")

    return running_loss / len(dataloader.dataset)


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
        DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED
    )

    encoder, decoder, autoencoder = build_autoencoder(latent_dim=LATENT_DIM)
    autoencoder = autoencoder.to(device)

    criterion = CombinedLoss(
        mse_weight=MSE_WEIGHT,
        ssim_weight=SSIM_WEIGHT,
        data_range=1.0
    ).to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(autoencoder.state_dict())
    early_stop_patience = 7
    epochs_no_improve = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        train_loss = train_one_epoch(autoencoder, train_loader, criterion, optimizer, device)
        val_loss = validate_one_epoch(autoencoder, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss : {train_loss:.6f}")
        print(f"Val   Loss : {val_loss:.6f}")
        print(f"LR         : {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(autoencoder.state_dict())
            torch.save(autoencoder.state_dict(), MODELS_DIR / "best_autoencoder.pth")
            epochs_no_improve = 0
            print("✓ Best model saved.")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

    autoencoder.load_state_dict(best_model_weights)
    torch.save(autoencoder.state_dict(), MODELS_DIR / "final_autoencoder1.pth")
    torch.save(autoencoder.encoder.state_dict(), MODELS_DIR / "encoder_only1.pth")
    torch.save(autoencoder.decoder.state_dict(), MODELS_DIR / "decoder_only1.pth")

    test_loss = evaluate(autoencoder, test_loader, criterion, device)
    print(f"\nTest Combined Loss: {test_loss:.6f}")

    plot_history(history, save_path=str(PLOTS_DIR / "ae_loss_curve.png"))
    show_reconstructions(
        autoencoder, test_loader, device, class_names=class_names,
        save_path=str(PLOTS_DIR / "ae_reconstructions.png"),
        num_images=6, skip_nothing=True,
    )