"""
Evaluate CNN and AE+MLP pipelines against a real-world ASL image dataset.

Expected dataset layout:
    dataset_root/
        A/  img1.jpg  img2.jpg  ...
        B/  ...
        0/  ...
        ...

Usage:
    python evaluate_realworld.py --dataset /path/to/dataset
    python evaluate_realworld.py --dataset /path/to/dataset --output results.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# ---------------------------------------------------------------------------
# Paths (relative to this file)
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
MODELS_DIR = HERE / "models"

CNN_PATH     = MODELS_DIR / "best_cnn_asl39.pth"
ENCODER_PATH = MODELS_DIR / "encoder_ae_asl39_ld256.pth"
MLP_PATH     = MODELS_DIR / "mlp_classifier.pkl"

LATENT_DIM = 256
NUM_CLASSES = 39

CLASS_NAMES = [
    '0','1','2','3','4','5','6','7','8','9',
    'A','B','C','D','E','F','G','H','I','J','K','L','M',
    'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
    'nothing','space','unknown',
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    return preprocess(img)   # (1, 64, 64)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------
def load_cnn(device: torch.device) -> torch.nn.Module:
    from cnn_arch import build_classifier
    model = build_classifier(num_classes=NUM_CLASSES, head="standard").to(device)
    model.load_state_dict(torch.load(CNN_PATH, map_location=device, weights_only=True))
    model.eval()
    return model


def load_encoder(device: torch.device) -> torch.nn.Module:
    from autoencoder_arch import Encoder
    enc = Encoder(latent_dim=LATENT_DIM).to(device)
    enc.load_state_dict(torch.load(ENCODER_PATH, map_location=device, weights_only=True))
    enc.eval()
    return enc


def load_mlp():
    import joblib
    return joblib.load(MLP_PATH)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict_cnn(model, tensor: torch.Tensor, device: torch.device) -> int:
    logits = model(tensor.unsqueeze(0).to(device))
    return int(logits.argmax(dim=1).item())


@torch.no_grad()
def encode(encoder, tensor: torch.Tensor, device: torch.device) -> np.ndarray:
    z = encoder(tensor.unsqueeze(0).to(device))
    return z.cpu().numpy()


def predict_mlp(mlp, z: np.ndarray) -> int:
    label = mlp.predict(z)[0]
    # labels may be string class names or integer indices
    if isinstance(label, (int, np.integer)):
        return int(label)
    return CLASS_TO_IDX.get(str(label), -1)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_samples(dataset_root: Path):
    """Return list of (path, class_idx) for every image found under dataset_root/ClassName/."""
    samples = []
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in CLASS_TO_IDX:
            print(f"  [skip] unknown class folder: {class_name!r}")
            continue
        idx = CLASS_TO_IDX[class_name]
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() in IMG_EXTS:
                samples.append((img_path, idx))
    return samples


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------
def evaluate(samples, cnn_model, encoder, mlp, device, batch_size=64):
    n = len(samples)
    cnn_correct = 0
    mlp_correct = 0 if mlp is not None else None

    per_class_cnn  = {i: [0, 0] for i in range(NUM_CLASSES)}  # [correct, total]
    per_class_mlp  = {i: [0, 0] for i in range(NUM_CLASSES)} if mlp is not None else None

    for i, (path, true_idx) in enumerate(samples):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  {i+1}/{n} ...", end="\r")

        try:
            tensor = load_image(path)
        except Exception as e:
            print(f"\n  [warn] could not load {path}: {e}")
            continue

        pred_cnn = predict_cnn(cnn_model, tensor, device)
        per_class_cnn[true_idx][1] += 1
        if pred_cnn == true_idx:
            cnn_correct += 1
            per_class_cnn[true_idx][0] += 1

        if mlp is not None:
            z = encode(encoder, tensor, device)
            pred_mlp = predict_mlp(mlp, z)
            per_class_mlp[true_idx][1] += 1
            if pred_mlp == true_idx:
                mlp_correct += 1
                per_class_mlp[true_idx][0] += 1

    print()
    return cnn_correct, mlp_correct, per_class_cnn, per_class_mlp, n


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_report(cnn_correct, mlp_correct, per_class_cnn, per_class_mlp, n):
    cnn_acc = cnn_correct / n * 100
    print(f"\n{'='*50}")
    print(f"  Total images evaluated : {n}")
    print(f"  CNN accuracy           : {cnn_acc:.2f}%  ({cnn_correct}/{n})")
    if mlp_correct is not None:
        mlp_acc = mlp_correct / n * 100
        print(f"  AE+MLP accuracy        : {mlp_acc:.2f}%  ({mlp_correct}/{n})")
    print(f"{'='*50}\n")

    print(f"{'Class':<10} {'CNN acc':>10}", end="")
    if mlp_correct is not None:
        print(f" {'AE+MLP acc':>12}", end="")
    print()
    print("-" * (23 + (13 if mlp_correct is not None else 0)))

    for idx, name in enumerate(CLASS_NAMES):
        c, t = per_class_cnn[idx]
        if t == 0:
            continue
        row = f"{name:<10} {c/t*100:>9.1f}%"
        if mlp_correct is not None and per_class_mlp is not None:
            mc, mt = per_class_mlp[idx]
            row += f"  {mc/mt*100:>10.1f}%" if mt > 0 else f"  {'N/A':>10}"
        print(row)


def save_csv(cnn_correct, mlp_correct, per_class_cnn, per_class_mlp, n, output_path: Path):
    import csv
    rows = [["class", "total", "cnn_correct", "cnn_acc_pct"]]
    if mlp_correct is not None:
        rows[0] += ["mlp_correct", "mlp_acc_pct"]

    for idx, name in enumerate(CLASS_NAMES):
        c, t = per_class_cnn[idx]
        if t == 0:
            continue
        row = [name, t, c, f"{c/t*100:.2f}"]
        if mlp_correct is not None and per_class_mlp is not None:
            mc, mt = per_class_mlp[idx]
            row += [mc, f"{mc/mt*100:.2f}" if mt > 0 else "N/A"]
        rows.append(row)

    # summary row
    summary = ["TOTAL", n, cnn_correct, f"{cnn_correct/n*100:.2f}"]
    if mlp_correct is not None:
        summary += [mlp_correct, f"{mlp_correct/n*100:.2f}"]
    rows.append(summary)

    with open(output_path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"Results saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-world ASL evaluation")
    parser.add_argument("--dataset", required=True, help="Path to dataset root (class-name subfolders)")
    parser.add_argument("--output",  default="realworld_results.csv", help="Output CSV path")
    parser.add_argument("--no-mlp",  action="store_true", help="Skip AE+MLP (CNN only)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset)
    if not dataset_root.is_dir():
        sys.exit(f"Dataset path not found: {dataset_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Loading CNN...")
    cnn_model = load_cnn(device)

    encoder = None
    mlp = None
    if not args.no_mlp:
        if not MLP_PATH.exists():
            print(f"[warn] MLP not found at {MLP_PATH} — running CNN only.")
            print("       Ask your teammate to add joblib.dump(model, 'mlp_classifier.pkl') to mlp_asl.py.")
        else:
            print("Loading AE encoder...")
            encoder = load_encoder(device)
            print("Loading MLP...")
            mlp = load_mlp()

    print(f"Scanning dataset: {dataset_root}")
    samples = collect_samples(dataset_root)
    if not samples:
        sys.exit("No images found. Check dataset path and folder structure (dataset/ClassName/img.jpg).")
    print(f"Found {len(samples)} images across {len({s[1] for s in samples})} classes.\n")

    cnn_correct, mlp_correct, per_class_cnn, per_class_mlp, n = evaluate(
        samples, cnn_model, encoder, mlp, device
    )

    print_report(cnn_correct, mlp_correct, per_class_cnn, per_class_mlp, n)
    save_csv(cnn_correct, mlp_correct, per_class_cnn, per_class_mlp, n, Path(args.output))


if __name__ == "__main__":
    main()
