"""
Evaluate the best ASL39 CNN on the ayuraj/asl-dataset Kaggle dataset.

Usage:
    python evaluate_kaggle_asl.py
    python evaluate_kaggle_asl.py --output my_results.csv
"""
import argparse
import sys
from pathlib import Path

import kagglehub
import torch
from torchvision import transforms
from PIL import Image

HERE = Path(__file__).parent

# best model by val_acc from ablation (ab5_cosine and ab4_aug_ls tied at 0.9997;
# ab5 chosen as the final step in the ablation chain)
CNN_MODEL = HERE / "cnn_classifier" / "models" / "best_cnn_classifier_asl39_ab5_cosine.pth"
CNN_HEAD  = "standard"

sys.path.insert(0, str(HERE / "cnn_classifier"))
from model import build_classifier  # noqa: E402

NUM_CLASSES = 39
CLASS_NAMES = [
    "0","1","2","3","4","5","6","7","8","9",
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "nothing","space","unknown",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# RGB → grayscale → 64×64 → [0,1]  (matches training pipeline exactly)
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def find_dataset_root(base: Path) -> Path:
    """Walk one level deep to find the folder whose children are class dirs."""
    for candidate in [base] + [p for p in sorted(base.iterdir()) if p.is_dir()]:
        if any(d.is_dir() and d.name.upper() in CLASS_TO_IDX for d in candidate.iterdir()):
            return candidate
    return base  # fallback


def collect_samples(dataset_root: Path):
    samples, skipped = [], []
    for class_dir in sorted(dataset_root.iterdir()):
        if not class_dir.is_dir():
            continue
        name = class_dir.name.upper()
        if name not in CLASS_TO_IDX:
            skipped.append(class_dir.name)
            continue
        idx = CLASS_TO_IDX[name]
        for p in class_dir.iterdir():
            if p.suffix.lower() in IMG_EXTS:
                samples.append((p, idx))
    if skipped:
        print(f"  [skip] unrecognised folders: {skipped}")
    return samples


def load_image(path: Path) -> torch.Tensor:
    return preprocess(Image.open(path).convert("RGB"))  # (1, 64, 64)


@torch.no_grad()
def evaluate(model, samples, device):
    model.eval()
    n = len(samples)
    correct = 0
    per_class = {i: [0, 0] for i in range(NUM_CLASSES)}  # [correct, total]

    for i, (img_path, true_idx) in enumerate(samples):
        if i % 500 == 0:
            print(f"  {i}/{n} ...", end="\r", flush=True)
        try:
            tensor = load_image(img_path)
        except Exception as e:
            print(f"\n  [warn] could not load {img_path}: {e}")
            continue
        pred = int(model(tensor.unsqueeze(0).to(device)).argmax(1).item())
        per_class[true_idx][1] += 1
        if pred == true_idx:
            correct += 1
            per_class[true_idx][0] += 1

    print(f"  {n}/{n} ... done")
    return correct, per_class, n


def print_report(correct, per_class, n):
    print(f"\n{'='*46}")
    print(f"  Total images : {n}")
    print(f"  CNN accuracy : {correct/n*100:.2f}%  ({correct}/{n})")
    print(f"{'='*46}")
    print(f"\n{'Class':<12} {'Correct':>8} {'Total':>8} {'Acc':>8}")
    print("-" * 40)
    for idx, name in enumerate(CLASS_NAMES):
        c, t = per_class[idx]
        if t == 0:
            continue
        print(f"{name:<12} {c:>8} {t:>8} {c/t*100:>7.1f}%")


def save_csv(correct, per_class, n, path: Path):
    import csv
    rows = [["class", "total", "correct", "acc_pct"]]
    for idx, name in enumerate(CLASS_NAMES):
        c, t = per_class[idx]
        if t == 0:
            continue
        rows.append([name, t, c, f"{c/t*100:.2f}"])
    rows.append(["TOTAL", n, correct, f"{correct/n*100:.2f}"])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"\nResults saved -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kaggle-dataset", default="ayuraj/asl-dataset",
                        help="Kaggle dataset id, e.g. owner/dataset-name")
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: realworld/<dataset-name>_results.csv)")
    args = parser.parse_args()

    dataset_slug = args.kaggle_dataset.split("/")[-1]
    output_path = Path(args.output) if args.output else Path(f"realworld/{dataset_slug}_results.csv")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Downloading dataset via kagglehub ...")
    dataset_path = Path(kagglehub.dataset_download(args.kaggle_dataset))
    print(f"Dataset path: {dataset_path}")

    dataset_root = find_dataset_root(dataset_path)
    print(f"Dataset root: {dataset_root}")

    samples = collect_samples(dataset_root)
    if not samples:
        sys.exit("No samples found — check dataset folder structure.")
    num_found_classes = len({s[1] for s in samples})
    print(f"Found {len(samples)} images across {num_found_classes} classes.\n")

    print(f"Loading model: {CNN_MODEL.name}")
    model = build_classifier(num_classes=NUM_CLASSES, head=CNN_HEAD).to(device)
    model.load_state_dict(torch.load(CNN_MODEL, map_location=device, weights_only=True))
    model.eval()

    print("Evaluating ...")
    correct, per_class, n = evaluate(model, samples, device)

    print_report(correct, per_class, n)
    save_csv(correct, per_class, n, output_path)


if __name__ == "__main__":
    main()
