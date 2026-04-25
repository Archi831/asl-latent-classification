"""
Zero-shot cross-dataset evaluation.

Evaluates each of the three CNN variants (ab5/standard, ab6/shallow, ab7/deep)
directly on ayuraj and danrasband — no fine-tuning, weights as trained on ASL39.

Usage:
    python evaluate_zeroshot_crosseval.py
"""
import csv
import sys
from pathlib import Path

import kagglehub
import torch
from torchvision import transforms
from PIL import Image

HERE = Path(__file__).parent
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

MODELS = [
    ("ab5", HERE / "cnn_classifier/models/best_cnn_classifier_asl39_ab5_cosine.pth",      "standard"),
    ("ab6", HERE / "cnn_classifier/models/best_cnn_classifier_asl39_ab6_head_shallow.pth", "shallow"),
    ("ab7", HERE / "cnn_classifier/models/best_cnn_classifier_asl39_ab7_head_deep.pth",    "deep"),
]

DATASETS = {
    "ayuraj":     "ayuraj/asl-dataset",
    "danrasband": "danrasband/asl-alphabet-test",
}

OUT_DIR = HERE / "realworld"

eval_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


def has_class_dirs(path: Path) -> bool:
    try:
        return any(d.is_dir() and d.name.upper() in CLASS_TO_IDX for d in path.iterdir())
    except Exception:
        return False


def find_dataset_root(base: Path) -> Path:
    queue = [base]
    for _ in range(3):
        next_q = []
        for p in queue:
            if has_class_dirs(p):
                return p
            try:
                next_q.extend(d for d in sorted(p.iterdir()) if d.is_dir())
            except Exception:
                pass
        queue = next_q
    return base


def collect_samples(root: Path) -> list:
    samples, skipped = [], []
    for class_dir in sorted(root.iterdir()):
        if not class_dir.is_dir():
            continue
        name = class_dir.name.upper()
        if name not in CLASS_TO_IDX:
            skipped.append(class_dir.name)
            continue
        idx = CLASS_TO_IDX[name]
        for p in class_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                samples.append((p, idx))
    if skipped:
        print(f"  [skip] {skipped}")
    return samples


def download(ds_name: str, kaggle_id: str) -> list:
    print(f"  Downloading {kaggle_id} ...")
    base = Path(kagglehub.dataset_download(kaggle_id))
    root = find_dataset_root(base)
    samples = collect_samples(root)
    n_cls = len({s[1] for s in samples})
    print(f"  [{ds_name}] {len(samples)} images, {n_cls} classes  (root: {root.name})")
    return samples


@torch.no_grad()
def evaluate(model, samples, device):
    model.eval()
    n = len(samples)
    correct = 0
    per_class = {i: [0, 0] for i in range(NUM_CLASSES)}

    for i, (img_path, true_idx) in enumerate(samples):
        if i % 500 == 0:
            print(f"  {i}/{n} ...", end="\r", flush=True)
        try:
            img = eval_tf(Image.open(img_path).convert("RGB"))
        except Exception as e:
            print(f"\n  [warn] {img_path}: {e}")
            continue
        pred = int(model(img.unsqueeze(0).to(device)).argmax(1).item())
        per_class[true_idx][1] += 1
        if pred == true_idx:
            correct += 1
            per_class[true_idx][0] += 1

    print(f"  {n}/{n} done      ")
    return correct, per_class, n


def save_csv(correct, per_class, n, path: Path):
    rows = [["class", "total", "correct", "acc_pct"]]
    for idx, name in enumerate(CLASS_NAMES):
        c, t = per_class[idx]
        if t == 0:
            continue
        rows.append([name, t, c, f"{c/t*100:.2f}"])
    rows.append(["TOTAL", n, correct, f"{correct/n*100:.2f}"])
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"  Saved -> {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    print("=== Downloading datasets ===")
    all_samples = {name: download(name, kid) for name, kid in DATASETS.items()}
    print()

    summary_rows = [["model_tag", "head", "dataset", "n", "accuracy_pct"]]

    for model_tag, model_path, head in MODELS:
        print(f"\n{'='*55}")
        print(f"  MODEL: {model_tag}  head={head}")
        print(f"{'='*55}")
        model = build_classifier(num_classes=NUM_CLASSES, head=head).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        for ds_name, samples in all_samples.items():
            print(f"\n  Evaluating on {ds_name} ...")
            correct, per_class, n = evaluate(model, samples, device)
            acc = correct / n * 100
            print(f"  {model_tag} on {ds_name}: {acc:.2f}%  ({correct}/{n})")
            out = OUT_DIR / f"zeroshot_{model_tag}_{ds_name}.csv"
            save_csv(correct, per_class, n, out)
            summary_rows.append([model_tag, head, ds_name, n, f"{acc:.2f}"])

    print(f"\n\n{'='*55}")
    print("  SUMMARY")
    print(f"{'='*55}")
    print(f"{'Model':<6} {'Head':<10} {'Dataset':<12} {'N':>6} {'Acc':>8}")
    print("-" * 48)
    for row in summary_rows[1:]:
        print(f"{row[0]:<6} {row[1]:<10} {row[2]:<12} {row[3]:>6} {row[4]:>7}%")

    summary_path = OUT_DIR / "zeroshot_summary.csv"
    with open(summary_path, "w", newline="") as f:
        csv.writer(f).writerows(summary_rows)
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
