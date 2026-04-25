"""
Cross-dataset fine-tuning experiment.

Runs two experiments:
  1. Fine-tune on danrasband/asl-alphabet-test  → evaluate on ayuraj/asl-dataset
  2. Fine-tune on ayuraj/asl-dataset            → evaluate on danrasband/asl-alphabet-test

Usage:
    python finetune_crosseval.py
"""
import copy
import sys
import time
from pathlib import Path

import kagglehub
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

HERE = Path(__file__).parent

MODELS = [
    ("ab5", HERE / "cnn_classifier" / "models" / "best_cnn_classifier_asl39_ab5_cosine.pth",      "standard"),
    ("ab6", HERE / "cnn_classifier" / "models" / "best_cnn_classifier_asl39_ab6_head_shallow.pth", "shallow"),
    ("ab7", HERE / "cnn_classifier" / "models" / "best_cnn_classifier_asl39_ab7_head_deep.pth",    "deep"),
]

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

eval_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

train_tf = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=8),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.08)),
])


# ── dataset ───────────────────────────────────────────────────────────────────

class ImageSamples(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def find_dataset_root(base: Path) -> Path:
    for candidate in [base] + [p for p in sorted(base.iterdir()) if p.is_dir()]:
        if any(d.is_dir() and d.name.upper() in CLASS_TO_IDX for d in candidate.iterdir()):
            return candidate
    return base


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
        print(f"  [skip] {skipped}")
    return samples


def download(dataset_id: str) -> list:
    print(f"  Downloading {dataset_id} ...")
    path = Path(kagglehub.dataset_download(dataset_id))
    root = find_dataset_root(path)
    samples = collect_samples(root)
    classes = len({s[1] for s in samples})
    print(f"  Found {len(samples)} images across {classes} classes  (root: {root.name})")
    return samples


# ── model ─────────────────────────────────────────────────────────────────────

def load_model(device, model_path, head):
    model = build_classifier(num_classes=NUM_CLASSES, head=head).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    return model


def freeze_early_blocks(model):
    """Freeze first two ConvBlocks; unfreeze last two blocks + classifier."""
    for param in model.parameters():
        param.requires_grad = False
    # features is Sequential: [CB0, Pool, CB1, Pool, CB2, Pool, CB3, Pool, AvgPool, Flatten]
    # indices 4,5 = ConvBlock(64,128)+Pool, 6,7 = ConvBlock(128,256)+Pool
    for layer in list(model.features.children())[4:]:
        for param in layer.parameters():
            param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}")


# ── fine-tuning ───────────────────────────────────────────────────────────────

def finetune(model, samples, device, epochs=30, batch_size=32, lr=1e-4, val_split=0.2):
    n_val  = max(1, int(len(samples) * val_split))
    n_train = len(samples) - n_val
    train_set, val_set = random_split(
        ImageSamples(samples, train_tf),
        [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    # val uses eval transforms — rebuild with eval_tf
    val_samples_list = [samples[i] for i in val_set.indices]
    val_dataset = ImageSamples(val_samples_list, eval_tf)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=1e-4
    )
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state   = copy.deepcopy(model.state_dict())
    patience, no_improve = 8, 0

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            criterion(model(imgs), labels).backward()
            optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                correct += (model(imgs).argmax(1) == labels).sum().item()
                total   += labels.size(0)
        val_acc = correct / total

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state   = copy.deepcopy(model.state_dict())
            no_improve   = 0
        else:
            no_improve += 1

        marker = " *" if improved else ""
        print(f"  epoch {epoch:2d}/{epochs}  val_acc={val_acc:.4f}{marker}")

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    elapsed = time.time() - t0
    print(f"  Fine-tuning done in {elapsed:.1f}s  |  best val_acc={best_val_acc:.4f}")
    model.load_state_dict(best_state)
    return model


# ── evaluation ────────────────────────────────────────────────────────────────

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

    print(f"  {n}/{n} ... done      ")
    return correct, per_class, n


def print_report(label, correct, per_class, n):
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Total images : {n}")
    print(f"  CNN accuracy : {correct/n*100:.2f}%  ({correct}/{n})")
    print(f"{'='*50}")
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
    print(f"  Saved -> {path}")


# ── main ──────────────────────────────────────────────────────────────────────

DANRASBAND = "danrasband/asl-alphabet-test"
AYURAJ     = "ayuraj/asl-dataset"

def run_experiment(train_id, test_id, model_tag, model_path, head, label, out_path, device):
    print(f"\n{'#'*60}")
    print(f"  EXPERIMENT [{model_tag}]: {label}")
    print(f"{'#'*60}")

    print("\n[1] Downloading datasets ...")
    train_samples = download(train_id)
    test_samples  = download(test_id)

    print(f"\n[2] Loading base model ({model_tag}, head={head}) ...")
    model = load_model(device, model_path, head)
    freeze_early_blocks(model)

    print("\n[3] Fine-tuning ...")
    model = finetune(model, train_samples, device)
    torch.save(model.state_dict(),
               Path(f"realworld/finetuned_{model_tag}_{train_id.split('/')[-1]}.pth"))

    print("\n[4] Evaluating on test dataset ...")
    correct, per_class, n = evaluate(model, test_samples, device)
    print_report(label, correct, per_class, n)
    save_csv(correct, per_class, n, out_path)
    return correct / n


if __name__ == "__main__":
    import csv

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t_total = time.time()

    EXPERIMENTS = [
        (DANRASBAND, AYURAJ,     "Fine-tuned on danrasband -> tested on ayuraj",    "danrasband_test_ayuraj"),
        (AYURAJ,     DANRASBAND, "Fine-tuned on ayuraj -> tested on danrasband",    "ayuraj_test_danrasband"),
    ]

    summary_rows = [["model_tag", "head", "experiment", "accuracy_pct"]]

    for model_tag, model_path, head in MODELS:
        print(f"\n\n{'='*60}")
        print(f"  MODEL: {model_tag}  head={head}")
        print(f"{'='*60}")
        for train_id, test_id, label, exp_slug in EXPERIMENTS:
            out_path = Path(f"realworld/finetune_{model_tag}_{exp_slug}.csv")
            acc = run_experiment(
                train_id=train_id, test_id=test_id,
                model_tag=model_tag, model_path=model_path, head=head,
                label=label, out_path=out_path, device=device,
            )
            summary_rows.append([model_tag, head, exp_slug, f"{acc*100:.2f}"])

    print(f"\n\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<6} {'Head':<10} {'Experiment':<35} {'Acc':>8}")
    print("-" * 60)
    for row in summary_rows[1:]:
        print(f"{row[0]:<6} {row[1]:<10} {row[2]:<35} {row[3]:>7}%")

    summary_path = Path("realworld/finetune_crosseval_summary.csv")
    with open(summary_path, "w", newline="") as f:
        csv.writer(f).writerows(summary_rows)
    print(f"\nSummary saved -> {summary_path}")
    print(f"Total wall time: {time.time() - t_total:.1f}s")
