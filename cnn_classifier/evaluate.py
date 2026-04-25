import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "autoencoder"))

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from data_prep import load_preprocessed_datasets, set_seed
from model import build_classifier

DATA_DIR   = str(Path(__file__).resolve().parent.parent / "data" / "asl_alphabet_train" / "asl_alphabet_train")
SEED       = 67
MODELS_DIR = Path(__file__).parent / "models"
PLOTS_DIR  = Path(__file__).parent / "outputs" / "plots"

def get_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            preds  = model(images).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.title("CNN Classifier — Confusion Matrix (Test Set)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")

if __name__ == "__main__":
    set_seed(SEED)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_loader, class_names = load_preprocessed_datasets(DATA_DIR, batch_size=128, seed=SEED)

    model = build_classifier().to(device)
    model.load_state_dict(torch.load(MODELS_DIR / "best_cnn_classifier.pth", map_location=device))

    labels, preds = get_predictions(model, test_loader, device)
    acc = accuracy_score(labels, preds)
    print(f"\nTest Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    plot_confusion_matrix(labels, preds, class_names, PLOTS_DIR / "cnn_confusion_matrix.png")
