from pathlib import Path
import copy
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
NUM_CLASSES   = 29
EPOCHS        = 50
LEARNING_RATE = 1e-3

# Reconstruction loss weights (MSE + SSIM)
SSIM_WEIGHT = 0.5
MSE_WEIGHT  = 0.5

# How much to weight the classification loss vs the reconstruction loss.
# 0.4 means: total = 0.6 * recon_loss + 0.4 * cls_loss
# Increase towards 0.6–0.7 if MLP accuracy matters more than reconstruction quality.
CLS_WEIGHT   = 0.4
RECON_WEIGHT = 1.0 - CLS_WEIGHT

# Label smoothing prevents the classifier head from becoming overconfident,
# which keeps embeddings from collapsing into tight but non-generalising clusters
LABEL_SMOOTHING = 0.1

MODELS_DIR = Path("models")
PLOTS_DIR  = Path("outputs/plots")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class ReconstructionLoss(nn.Module):
    """Weighted combination of MSE and SSIM, same as before."""
    def __init__(self, mse_weight: float = 0.5, ssim_weight: float = 0.5, data_range: float = 1.0):
        super().__init__()
        self.mse_weight  = mse_weight
        self.ssim_weight = ssim_weight
        self.mse         = nn.MSELoss()
        self.ssim        = StructuralSimilarityIndexMeasure(data_range=data_range)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.mse_weight * self.mse(pred, target) + \
               self.ssim_weight * (1.0 - self.ssim(pred, target))


def plot_history(history, save_path="outputs/plots/ae_loss_curve.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["train_loss"], label="Train")
    ax1.plot(history["val_loss"],   label="Val")
    ax1.set_title("Total Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["train_cls_loss"], label="Train")
    ax2.plot(history["val_cls_loss"],   label="Val")
    ax2.set_title("Classification Loss (head on z)")
    ax2.set_xlabel("Epoch")
    ax2.legend()

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
        x_hat, _ = model(images)   # unpack (recon, logits)

    images = images.cpu()
    x_hat  = x_hat.cpu()

    plt.figure(figsize=(12, 4))
    for i in range(len(images)):
        plt.subplot(2, len(images), i + 1)
        plt.imshow(images[i].squeeze(0), cmap="gray")
        plt.title(f"Original\n{class_names[selected_labels[i]]}")
        plt.axis("off")

        plt.subplot(2, len(images), i + 1 + len(images))
        plt.imshow(x_hat[i].squeeze(0), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


def train_one_epoch(model, dataloader, recon_criterion, cls_criterion, optimizer, device):
    model.train()
    total_loss     = 0.0
    total_cls_loss = 0.0
    loop = tqdm(dataloader, desc="Training", leave=False)

    for images, labels in loop:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        x_hat, logits = model(images)

        recon_loss = recon_criterion(x_hat, images)
        cls_loss   = cls_criterion(logits, labels)
        loss       = RECON_WEIGHT * recon_loss + CLS_WEIGHT * cls_loss

        loss.backward()
        optimizer.step()

        total_loss     += loss.item()     * images.size(0)
        total_cls_loss += cls_loss.item() * images.size(0)
        loop.set_postfix(loss=f"{loss.item():.5f}", cls=f"{cls_loss.item():.5f}")

    n = len(dataloader.dataset)
    return total_loss / n, total_cls_loss / n


def validate_one_epoch(model, dataloader, recon_criterion, cls_criterion, device):
    model.eval()
    total_loss     = 0.0
    total_cls_loss = 0.0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Validation", leave=False)
        for images, labels in loop:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            x_hat, logits = model(images)
            recon_loss = recon_criterion(x_hat, images)
            cls_loss   = cls_criterion(logits, labels)
            loss       = RECON_WEIGHT * recon_loss + CLS_WEIGHT * cls_loss

            total_loss     += loss.item()     * images.size(0)
            total_cls_loss += cls_loss.item() * images.size(0)
            loop.set_postfix(loss=f"{loss.item():.5f}", cls=f"{cls_loss.item():.5f}")

    n = len(dataloader.dataset)
    return total_loss / n, total_cls_loss / n


def evaluate(model, dataloader, recon_criterion, cls_criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            x_hat, logits = model(images)
            recon_loss = recon_criterion(x_hat, images)
            cls_loss   = cls_criterion(logits, labels)
            total_loss += (RECON_WEIGHT * recon_loss + CLS_WEIGHT * cls_loss).item() * images.size(0)

    return total_loss / len(dataloader.dataset)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, test_loader, class_names = load_preprocessed_datasets(
        DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE, seed=SEED
    )

    encoder, decoder, autoencoder = build_autoencoder(
        latent_dim=LATENT_DIM, num_classes=NUM_CLASSES
    )
    autoencoder = autoencoder.to(device)

    recon_criterion = ReconstructionLoss(
        mse_weight=MSE_WEIGHT, ssim_weight=SSIM_WEIGHT, data_range=1.0
    ).to(device)

    # Label smoothing stops the head from driving z into overconfident tight clusters
    cls_criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING).to(device)

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
    )

    history = {"train_loss": [], "val_loss": [], "train_cls_loss": [], "val_cls_loss": []}
    best_val_loss        = float("inf")
    best_model_weights   = copy.deepcopy(autoencoder.state_dict())
    early_stop_patience  = 7
    epochs_no_improve    = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        train_loss, train_cls = train_one_epoch(
            autoencoder, train_loader, recon_criterion, cls_criterion, optimizer, device
        )
        val_loss, val_cls = validate_one_epoch(
            autoencoder, val_loader, recon_criterion, cls_criterion, device
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_cls_loss"].append(train_cls)
        history["val_cls_loss"].append(val_cls)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Train Loss : {train_loss:.6f}  (cls: {train_cls:.6f})")
        print(f"Val   Loss : {val_loss:.6f}  (cls: {val_cls:.6f})")
        print(f"LR         : {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss      = val_loss
            best_model_weights = copy.deepcopy(autoencoder.state_dict())
            torch.save(autoencoder.state_dict(), MODELS_DIR / "best_autoencoder.pth")
            epochs_no_improve  = 0
            print("✓ Best model saved.")
        else:
            epochs_no_improve += 1
            print(f"  No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

    autoencoder.load_state_dict(best_model_weights)
    torch.save(autoencoder.state_dict(),         MODELS_DIR / "final_autoencoder.pth")
    torch.save(autoencoder.encoder.state_dict(), MODELS_DIR / "encoder_only.pth")
    torch.save(autoencoder.decoder.state_dict(), MODELS_DIR / "decoder_only.pth")

    test_loss = evaluate(autoencoder, test_loader, recon_criterion, cls_criterion, device)
    print(f"\nTest Combined Loss: {test_loss:.6f}")

    plot_history(history, save_path=str(PLOTS_DIR / "ae_loss_curve.png"))
    show_reconstructions(
        autoencoder, test_loader, device, class_names=class_names,
        save_path=str(PLOTS_DIR / "ae_reconstructions.png"),
        num_images=6, skip_nothing=True,
    )