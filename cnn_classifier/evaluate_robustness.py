import sys
import importlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "autoencoder"))

import torch
import numpy as np
from sklearn.metrics import accuracy_score

from model import build_classifier

DATA_DIR   = str(Path(__file__).resolve().parent.parent / "data" / "asl-alphabet-numbers" / "asl-numbers-alphabet-dataset")
MODEL_PATH = Path(__file__).parent / "models" / "best_cnn_classifier_asl39_ab5_cosine.pth"
SEED       = 67
BATCH_SIZE = 128


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


def load_test_split(module_name, data_dir):
    mod = importlib.import_module(module_name)
    if module_name == "data_prep_resolution":
        _, _, test_loader, class_names = mod.load_preprocessed_datasets(
            data_dir, img_size=(64, 64), low_res_size=(16, 16),
            batch_size=BATCH_SIZE, seed=SEED,
        )
    else:
        _, _, test_loader, class_names = mod.load_preprocessed_datasets(
            data_dir, img_size=(64, 64), batch_size=BATCH_SIZE, seed=SEED,
        )
    return test_loader, class_names


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Model  : {MODEL_PATH}")
    print(f"Data   : {DATA_DIR}\n")

    model = build_classifier(num_classes=39, head="standard").to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded.\n")

    configs = [
        ("Brightness  (jitter=0.5)",      "data_prep_bright"),
        ("Noise       (gaussian=0.25)",   "data_prep_noise"),
        ("Resolution  (16x16 -> 64x64)",  "data_prep_resolution"),
    ]

    print(f"{'Perturbation':<35} {'Test Acc':>10}  {'Samples':>8}")
    print("-" * 57)
    for label, mod_name in configs:
        test_loader, _ = load_test_split(mod_name, DATA_DIR)
        labels, preds  = get_predictions(model, test_loader, device)
        acc = accuracy_score(labels, preds)
        print(f"{label:<35} {acc*100:>9.2f}%  {len(test_loader.dataset):>8}")
